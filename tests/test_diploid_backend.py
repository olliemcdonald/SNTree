"""Independent likelihood and EM references for the fixed-diploid backend."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from ete4 import Tree
from scipy.special import logsumexp
from scipy.stats import binom

from sntree.cli import _parse_fp_mode
from sntree.io.io_preprocess import build_all
from sntree.likelihood import leaf_emissions as emissions_module
from sntree.likelihood.em_alpha_beta import em_alpha_beta
from sntree.likelihood.em_soft import em_soft
from sntree.likelihood.leaf_emissions import prepare_leaf_emissions
from sntree.likelihood.local_constant_cn import build_local_likelihood_cache, make_local_cached_scorer
from sntree.likelihood.loglik_all_snvs import loglik_all_snvs
from sntree.likelihood.locus_loglik_batch import locus_loglik_batch, leaf_inside_weights
from sntree.likelihood.quick_null_stats import quick_null_stats


@pytest.fixture
def data():
    tree = Tree('((a,b)x,(c,d)y,e)root;', parser=1)
    cells = ['a', 'b', 'c', 'd', 'e']
    for i, node in enumerate(tree.traverse()):
        node.props['CN_profile'] = {
            0: {'cn_tot': i % 6, 'cn_a': i % 6, 'cn_b': 0},
            1: {'cn_tot': (i + 2) % 6, 'cn_a': (i + 2) % 6, 'cn_b': 0},
        }
        if node.is_leaf:
            node.props['cell_id'] = node.name
    ks = np.array([[5, 4, 0, 0, 0], [0, 0, 6, 5, 0], [0, 0, 0, 0, 0],
                   [2, 0, 3, 0, 1], [0, 0, 0, 0, 0], [9, 10, 8, 7, 6]])
    ns = np.full_like(ks, 10)
    ns[4] = 0
    ns[2, 1] = 0
    ids = [f's{i}' for i in range(len(ks))]
    ref = pd.DataFrame(ns - ks, index=ids, columns=cells)
    alt = pd.DataFrame(ks, index=ids, columns=cells)
    with patch('sntree.io.io_preprocess.build_transition_model', side_effect=AssertionError('unused transitions')):
        ct, ds, transitions = build_all(
            tree, pd.DataFrame({'idx': [0, 1]}), None, ref, alt,
            pd.DataFrame({'snv': ids, 'cna_idx': [0, 1, 0, 1, 0, 1]}),
        )
    assert transitions is None
    assert ct.leaf_desc_mask is None
    # Deliberately make dataset column order differ from tree traversal order.
    perm = np.array([3, 0, 4, 1, 2])
    ds.ks, ds.ns, ds.leaf_names = ds.ks[:, perm], ds.ns[:, perm], ds.leaf_names[perm]
    ct.leaf_order = np.argsort(perm)
    return ct, ds


def reference(ct, ds, batch, alpha, beta, p0, fp, cell_perm=None):
    """Direct enumeration by named cells, using SciPy's binomial distribution."""
    ks, ns = ds.ks[batch], ds.ns[batch]
    if cell_perm is not None:
        ks, ns = ks[:, cell_perm], ns[:, cell_perm]
    columns = {name: i for i, name in enumerate(ds.leaf_names)}
    descendants = np.zeros((ct.n_nodes, ds.n_leaves), dtype=bool)
    fp_vaf = np.full_like(ks, fp if isinstance(fp, float) else 0., dtype=float)
    for u, node in enumerate(ct.idx_to_ete):
        for leaf in node.leaves():
            descendants[u, columns[leaf.name]] = True
        if node.is_leaf and fp == 'one_over_c':
            cn = np.array([node.props['CN_profile'][int(seg)]['cn_tot'] for seg in ds.seg[batch]])
            fp_vaf[:, columns[node.name]] = np.divide(1., cn, out=np.zeros(len(cn)), where=cn > 0)
    lp0, lp_half, lp_fp = binom.logpmf(ks, ns, p0), binom.logpmf(ks, ns, .5), binom.logpmf(ks, ns, fp_vaf)
    with np.errstate(divide='ignore', invalid='ignore'):
        absent = np.logaddexp(np.log(alpha) + lp_fp, np.log1p(-alpha) + lp0)
        present = np.logaddexp(np.log1p(-beta) + lp_half, np.log(beta) + lp0)
        q = np.exp(np.log(alpha) + lp_fp - absent)
        w = np.exp(np.log1p(-beta) + lp_half - present)
    scores = np.array([np.where(mask, present, absent).sum(axis=1) for mask in descendants])
    return scores, absent.sum(axis=1), q, w, descendants


@pytest.mark.parametrize('fp', ['one_over_c', .17, .5])
@pytest.mark.parametrize('alpha,beta,p0', [(0., 0., .001), (.12, .27, .001),
                                         (1., 0., .001), (0., 1., .001),
                                         (0., 0., 0.), (1., 1., 1.)])
def test_likelihood_matches_direct_enumeration(data, fp, alpha, beta, p0):
    ct, ds = data
    for batch in ds.snvs_by_seg.values():
        expected = reference(ct, ds, batch, alpha, beta, p0, fp)
        nodes, null = locus_loglik_batch(ct, ds, None, batch, alpha, beta, p0, fp)
        np.testing.assert_allclose(nodes, expected[0], atol=1e-11)
        np.testing.assert_allclose(null, expected[1], atol=1e-11)
        assert not np.isnan(nodes).any()
        for j, s in enumerate(batch):
            assert quick_null_stats(ct, ds, s, alpha, p0, fp)[0] == pytest.approx(null[j])


def test_permutation_moves_reads_between_fixed_tip_emissions(data):
    ct, ds = data
    perm = np.array([4, 2, 0, 1, 3])
    for batch in ds.snvs_by_seg.values():
        expected = reference(ct, ds, batch, .12, .27, .001, 'one_over_c', perm)
        actual = locus_loglik_batch(ct, ds, None, batch, .12, .27, .001, cell_perm=perm)
        np.testing.assert_allclose(actual[0], expected[0], atol=1e-11)
        np.testing.assert_allclose(actual[1], expected[1], atol=1e-11)


def test_cn_only_affects_false_positive_component(data):
    ct, ds = data
    batch = ds.snvs_by_seg[0]
    old = prepare_leaf_emissions(ct, ds, batch, .2, .3)
    scalar_old = locus_loglik_batch(ct, ds, None, batch, .2, .3, p1_fp_mode=.17)
    ct.CN[:] = 0
    new = prepare_leaf_emissions(ct, ds, batch, .2, .3)
    np.testing.assert_array_equal(old.present, new.present)
    assert not np.allclose(old.absent, new.absent)
    scalar_new = locus_loglik_batch(ct, ds, None, batch, .2, .3, p1_fp_mode=.17)
    np.testing.assert_allclose(scalar_old[0], scalar_new[0])
    np.testing.assert_allclose(scalar_old[1], scalar_new[1])


def test_branch_weight_accumulation_matches_named_descendants(data):
    ct, ds = data
    descendants = reference(ct, ds, [0], .1, .2, .001, .17)[4]
    rng = np.random.default_rng(12)
    weights = rng.random((ct.n_nodes, 7))
    np.testing.assert_allclose(leaf_inside_weights(ct, weights), weights.T @ descendants)


@pytest.mark.parametrize('fp', ['one_over_c', .17, .5])
def test_joint_soft_em_update_matches_independent_reference_and_reuses_components(data, fp):
    ct, ds = data
    alpha, beta, pi0 = .12, .27, .3
    scores, null, q, w, descendants = reference(ct, ds, np.arange(ds.n_snvs), alpha, beta, .001, fp)
    pi = np.arange(1, ct.n_nodes + 1, dtype=float)
    pi /= pi.sum()
    joint = np.vstack([np.log(pi0) + null, np.log1p(-pi0) + np.log(pi[:, None]) + scores])
    normalizer = logsumexp(joint, axis=0)
    gamma = np.exp(joint - normalizer)
    inside = gamma[1:].T @ descendants
    cov = ds.ns > 0
    expected_alpha = np.sum((1-inside) * q * cov) / np.sum((1-inside) * cov)
    expected_beta = 1 - np.sum(inside * w * cov) / np.sum(inside * cov)
    expected_pi = gamma[1:].sum(axis=1)
    expected_pi /= expected_pi.sum()
    with patch.object(emissions_module, 'logpmf_binom', wraps=emissions_module.logpmf_binom) as pmf:
        a, b, proportions, pnull, history = em_soft(
            ct, ds, None, alpha, beta, init_pi0=pi0, init_pi_b=pi,
            p1_fp_mode=fp, joint=True, max_iter=1, batch_size=100,
        )
    assert pmf.call_count == (4 if fp == .5 else 6)  # two segments, no extra M-step binomials
    np.testing.assert_allclose([a, b, pnull], [expected_alpha, expected_beta, gamma[0].mean()])
    np.testing.assert_allclose(proportions, expected_pi)
    assert history[0]['ll'] == pytest.approx(normalizer.sum())


@pytest.mark.parametrize('fp', ['one_over_c', .17])
def test_hard_em_and_ml_match_direct_map_and_error_update(data, fp):
    ct, ds = data
    alpha, beta, pi0 = .12, .27, .3
    scores, null, q, w, descendants = reference(ct, ds, np.arange(ds.n_snvs), alpha, beta, .001, fp)
    joint = np.vstack([np.log(pi0) + null, np.log((1-pi0)/ct.n_nodes) + scores])
    best = joint.argmax(axis=0)
    forced = (ds.ks.sum(axis=1) < 2) | ((ds.ks > 0).sum(axis=1) < 2)
    best[forced] = 0
    values = joint[best, np.arange(ds.n_snvs)]
    inside = np.array([descendants[u-1] if u else np.zeros(ds.n_leaves, dtype=bool) for u in best])
    cov = ds.ns > 0
    expected_alpha = np.sum((~inside) * q * cov) / np.sum((~inside) * cov)
    expected_beta = 1 - np.sum(inside * w * cov) / np.sum(inside * cov)
    a, b, placements, hist = em_alpha_beta(
        ct, ds, None, alpha, beta, pi0=pi0, p1_fp_mode=fp,
        max_iter=1, damp=0., batch_size=2,
    )
    np.testing.assert_allclose([a, b], [expected_alpha, expected_beta])
    assert hist[0]['ll'] == pytest.approx(values.sum())
    ml, total = loglik_all_snvs(ct, ds, None, alpha, beta, pi0=pi0, p1_fp_mode=fp, batch_size=2)
    assert total == pytest.approx(values.sum())
    for s, snv_id in enumerate(ds.snv_ids):
        expected_node = best[s]-1 if best[s] else None
        assert placements[snv_id][0] == ml[snv_id][0] == expected_node
        assert placements[snv_id][1] == pytest.approx(values[s])


def test_soft_em_batch_sizes_and_fixed_parameters(data):
    ct, ds = data
    a = em_soft(ct, ds, None, .12, .27, joint=True, max_iter=5, tol=0., batch_size=1)
    b = em_soft(ct, ds, None, .12, .27, joint=True, max_iter=5, tol=0., batch_size=100)
    np.testing.assert_allclose(a[:2], b[:2])
    np.testing.assert_allclose(a[2], b[2])
    assert a[3] == pytest.approx(b[3])
    assert np.all(np.diff([h['ll'] for h in a[4]]) >= -1e-9)
    fixed = em_soft(ct, ds, None, .12, .27, joint=False, max_iter=3)
    assert fixed[:2] == (.12, .27)


@pytest.mark.parametrize('fp', ['one_over_c', .17])
def test_refinement_uses_same_emissions(data, fp):
    ct, ds = data
    # Refinement is restricted to a constant-CN clade.
    ct.CN[:] = 4
    for node in ct.idx_to_ete:
        for profile in node.props['CN_profile'].values():
            profile['cn_tot'] = 4
    profile = {0: {'cn_tot': 4}, 1: {'cn_tot': 4}}
    delta, absent = build_local_likelihood_cache(ds.ks, ds.ns, ds.seg, profile, .12, .27, .001, fp)
    score = make_local_cached_scorer(delta, absent, ds.leaf_names)
    expected = reference(ct, ds, np.arange(ds.n_snvs), .12, .27, .001, fp)
    _, total = score(ct.idx_to_ete[0])
    assert total == pytest.approx(np.maximum(expected[0].max(axis=0), expected[1]).sum())


def test_empty_batch_and_unsupported_edge_factor(data):
    ct, ds = data
    nodes, null = locus_loglik_batch(ct, ds, None, [])
    assert nodes.shape == (ct.n_nodes, 0)
    assert null.shape == (0,)
    with pytest.raises(ValueError, match='edge factors'):
        locus_loglik_batch(ct, ds, None, [0], include_edge=True)
    with pytest.raises(ValueError, match='single CNA segment'):
        locus_loglik_batch(ct, ds, None, [0, 1])


@pytest.mark.parametrize('value,expected', [('one_over_c', 'one_over_c'), ('0.17', .17), ('0', 0.), ('1', 1.)])
def test_fp_cli_parser(value, expected):
    assert _parse_fp_mode(value) == expected


def test_fit_score_and_warm_start_preserve_false_positive_policy(data, tmp_path, monkeypatch):
    import pickle
    from types import SimpleNamespace
    from sntree.config import Config
    from sntree.workflow import soft_em as workflow
    from sntree.workflow import score as scoring

    ct, ds = data
    monkeypatch.setattr(workflow, 'load_structures', lambda *a: (ct, ds, None))
    monkeypatch.setattr(scoring, 'load_structures', lambda *a: (ct, ds, None))
    config = Config(alpha_init=.12, beta_init=.27, p0=.01, p1_fp_mode=.17,
                    soft_em_max_iter=2, batch_size=2)
    paths = SimpleNamespace(preprocessed_tree='synthetic.new')
    workflow.run_soft_em('S', str(tmp_path), config, paths, joint=True)
    saved_path = tmp_path / 'S/sntree/soft_em/soft_em_results.pkl'
    with saved_path.open('rb') as f:
        saved = pickle.load(f)
    assert saved['p1_fp_mode'] == .17
    assert saved['p0'] == .01
    assert saved['likelihood_model'] == 'fixed_diploid'

    result = scoring.run_score('S', str(tmp_path), Config(), paths, replicates=0)
    expected = workflow.score_placements(
        ct, ds, None, alpha=saved['alpha'], beta=saved['beta'],
        pi_b=saved['pi_b'], pi_0=saved['pi_0'], p0=.01, p1_fp_mode=.17,
    )
    pd.testing.assert_frame_equal(result['observed'], expected)

    # A CLI-style pass 2 uses the saved emissions even with default Config().
    next_config = Config(soft_em_max_iter=1, batch_size=2)
    with patch.object(workflow, 'em_soft', wraps=workflow.em_soft) as fit:
        workflow.run_soft_em('S', str(tmp_path), next_config, paths, joint=False,
                             output_subdir='soft_em_pass2', warm_start_pkl=str(saved_path))
    assert fit.call_args.kwargs['p1_fp_mode'] == .17
    assert fit.call_args.kwargs['p0'] == .01


@pytest.mark.parametrize('value', ['nan', 'inf', '-0.1', '1.1', 'unknown'])
def test_fp_cli_rejects_invalid_modes(value):
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_fp_mode(value)
