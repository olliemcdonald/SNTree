"""
Tests for per-SNV placement confidence (llr_null / llr_margin) and the
cell → tip permutation used to build an empirical null.

The core claim under test: with the same number of alt-carrying cells, a locus
whose alt cells form a clade scores far higher llr_null than one whose alt
cells are scattered across the tree — and permuting the cell → tip assignment
collapses that difference.
"""

import numpy as np
import pandas as pd
import pytest
from ete4 import Tree

from sntree.io.io_preprocess import build_all
from sntree.likelihood.locus_loglik_batch import locus_loglik_batch
from sntree.workflow.soft_em import (
    score_placements,
    compute_map_placements,
    SCORE_STATUS_OK,
    SCORE_STATUS_NO_MARGIN,
    SCORE_STATUS_NO_VALID_BRANCH,
)

N_SEGMENTS = 1
DEPTH = 20          # reads per cell per locus
ALT = 10            # alt reads in a carrier cell
ALPHA = 0.001       # FP rate
BETA = 0.05         # FN rate
P0 = 0.001

N_CELLS = 16
CELLS = [f"c{i}" for i in range(N_CELLS)]

# Loci: same number of alt cells (8), different phylogenetic arrangement.
CLADE_CELLS     = CELLS[:8]        # one clade — the ancestor of c0..c7
SCATTERED_CELLS = CELLS[::2]       # one per cherry, no clade anywhere
SISTER_CELLS    = CELLS[:2]        # a cherry


def _balanced_newick(cells):
    """Balanced binary tree over len(cells) (a power of two) leaves."""
    nodes = list(cells)
    while len(nodes) > 1:
        nodes = [
            f"({nodes[i]},{nodes[i + 1]})n{len(nodes) // 2}_{i // 2}"
            for i in range(0, len(nodes), 2)
        ]
    return nodes[0].rsplit(")", 1)[0] + ")root;"


def _toy_tree(cn_by_node=None):
    """
    Balanced 16-leaf tree with named internal nodes and a CN profile on every
    node.

    cn_by_node : dict node_name → cn_tot for segment 0, overriding the default
                 of 2 (used to exercise the CN=0 status paths).
    """
    # parser=1 reads internal labels as names, matching read_preprocessed_tree
    t = Tree(_balanced_newick(CELLS), parser=1)

    cn_by_node = cn_by_node or {}
    for node in t.traverse():
        cn = cn_by_node.get(node.name, 2)
        node.props["CN_profile"] = {
            seg: {"cn_tot": cn, "cn_a": cn - cn // 2, "cn_b": cn // 2}
            for seg in range(N_SEGMENTS)
        }
        if node.is_leaf:
            node.props["cell_id"] = node.name

    rows = [
        {
            "sample_id": node.name,
            "idx": seg,
            "cn_tot": node.props["CN_profile"][seg]["cn_tot"],
            "cn_a": node.props["CN_profile"][seg]["cn_a"],
            "cn_b": node.props["CN_profile"][seg]["cn_b"],
        }
        for node in t.traverse()
        for seg in range(N_SEGMENTS)
    ]
    return t, pd.DataFrame(rows)


def _toy_dataset(loci, cn_by_node=None):
    """
    loci : dict snv_id → list of alt-carrying cell names.

    Every cell is covered at DEPTH; carriers get ALT alt reads.
    """
    t, cna_profiles = _toy_tree(cn_by_node)

    snv_ids = list(loci)
    alt = pd.DataFrame(0, index=snv_ids, columns=CELLS, dtype=int)
    ref = pd.DataFrame(DEPTH, index=snv_ids, columns=CELLS, dtype=int)

    for snv_id, carriers in loci.items():
        alt.loc[snv_id, carriers] = ALT
        ref.loc[snv_id, carriers] = DEPTH - ALT

    snv_df = pd.DataFrame({"snv": snv_ids, "cna_idx": [0] * len(snv_ids)})

    cna_tree, snv_dataset, transitions = build_all(
        ete_tree=t,
        cna_profiles=cna_profiles,
        sample_mapping=None,
        ref_df=ref,
        alt_df=alt,
        snv_df=snv_df,
    )
    return cna_tree, snv_dataset, transitions


def _uniform_params(cna_tree):
    """Uniform branch prior, so nothing but the data drives the placement."""
    n = cna_tree.n_nodes
    return dict(
        alpha=ALPHA, beta=BETA,
        pi_b=np.full(n, 1.0 / n), pi_0=0.5, p0=P0,
    )


@pytest.fixture
def scored():
    loci = {
        "clade":     CLADE_CELLS,
        "scattered": SCATTERED_CELLS,
        "sister":    SISTER_CELLS,
        "empty":     [],
    }
    cna_tree, snv_dataset, transitions = _toy_dataset(loci)
    df = score_placements(
        cna_tree, snv_dataset, transitions,
        batch_size=2, **_uniform_params(cna_tree)
    )
    return cna_tree, snv_dataset, transitions, df


# ── Task 1: the LLRs ──────────────────────────────────────────────────────

def test_clade_concordant_locus_scores_far_above_a_scattered_one(scored):
    _, _, _, df = scored

    assert df.loc["clade", "llr_null"] > df.loc["scattered", "llr_null"] + 20
    assert df.loc["clade", "score_status"] == SCORE_STATUS_OK
    assert df.loc["scattered", "score_status"] == SCORE_STATUS_OK


def test_clade_locus_places_on_the_clade_ancestor(scored):
    _, _, _, df = scored

    assert df.loc["clade", "node"] == "n2_0"    # ancestor of c0..c7
    assert df.loc["sister", "node"] == "n8_0"   # the (c0,c1) cherry


def test_empty_locus_is_null_with_nonpositive_llr(scored):
    _, _, _, df = scored

    assert df.loc["empty", "node"] == "Null"
    assert df.loc["empty", "llr_null"] <= 0


def test_margin_is_nonnegative_and_finite_for_ok_loci(scored):
    _, _, _, df = scored

    ok = df[df["score_status"] == SCORE_STATUS_OK]
    assert len(ok) > 0
    assert np.isfinite(ok["llr_margin"]).all()
    assert (ok["llr_margin"] >= 0).all()


def test_margin_matches_a_full_sort(scored):
    """np.partition must give the same runner-up a full sort would."""
    cna_tree, snv_dataset, transitions, df = scored
    params = _uniform_params(cna_tree)

    log_pi_b = np.log(params["pi_b"])
    log_pi_0 = np.log(params["pi_0"])
    log_1mpi0 = np.log(1.0 - params["pi_0"])

    batch = np.arange(snv_dataset.n_snvs)
    logL_nodes, logL_null = locus_loglik_batch(
        cna_tree, snv_dataset, transitions, batch,
        alpha=params["alpha"], beta=params["beta"], p0=params["p0"],
    )
    branch = log_1mpi0 + log_pi_b[:, None] + logL_nodes
    null = log_pi_0 + logL_null

    for j, snv_id in enumerate(snv_dataset.snv_ids[batch]):
        ordered = np.sort(branch[:, j])[::-1]
        assert df.loc[snv_id, "llr_null"] == pytest.approx(ordered[0] - null[j])
        assert df.loc[snv_id, "llr_margin"] == pytest.approx(ordered[0] - ordered[1])


def test_map_node_column_matches_the_original_scalar_loop(scored):
    """
    The node column (and its "Null" sentinel) must be unchanged by the
    vectorisation — downstream R code reads it.
    """
    cna_tree, snv_dataset, transitions, df = scored
    params = _uniform_params(cna_tree)

    log_pi_b = np.log(params["pi_b"])
    log_pi_0 = np.log(params["pi_0"])
    log_1mpi0 = np.log(1.0 - params["pi_0"])

    batch = np.arange(snv_dataset.n_snvs)
    logL_nodes, logL_null = locus_loglik_batch(
        cna_tree, snv_dataset, transitions, batch,
        alpha=params["alpha"], beta=params["beta"], p0=params["p0"],
    )
    log_branch_score = log_1mpi0 + log_pi_b[:, None] + logL_nodes
    log_null_score = log_pi_0 + logL_null

    # Verbatim reference implementation of the pre-change inner loop
    reference = {}
    for j, snv_idx in enumerate(batch):
        snv_id = snv_dataset.snv_ids[snv_idx]
        branch_scores = log_branch_score[:, j]
        null_score = log_null_score[j]
        best_branch_idx = int(np.argmax(branch_scores))
        if branch_scores[best_branch_idx] > null_score:
            node_name = cna_tree.idx_to_ete[best_branch_idx].name
        else:
            node_name = "Null"
        reference[snv_id] = node_name

    assert df["node"].to_dict() == reference


def test_compute_map_placements_keeps_its_dict_contract(scored):
    cna_tree, snv_dataset, transitions, df = scored

    placements = compute_map_placements(
        cna_tree, snv_dataset, transitions, batch_size=2,
        **_uniform_params(cna_tree)
    )
    assert placements == df["node"].to_dict()


# ── Task 2: permutation ───────────────────────────────────────────────────

def test_cell_perm_none_matches_identity_permutation(scored):
    cna_tree, snv_dataset, transitions, _ = scored
    batch = np.arange(snv_dataset.n_snvs)

    base = locus_loglik_batch(
        cna_tree, snv_dataset, transitions, batch,
        alpha=ALPHA, beta=BETA, p0=P0,
    )
    identity = locus_loglik_batch(
        cna_tree, snv_dataset, transitions, batch,
        alpha=ALPHA, beta=BETA, p0=P0,
        cell_perm=np.arange(snv_dataset.n_leaves),
    )

    np.testing.assert_array_equal(base[0], identity[0])
    np.testing.assert_array_equal(base[1], identity[1])


def test_permutation_moves_the_reads_not_the_tree(scored):
    """
    Permuting the cell → tip assignment must give the same likelihoods as
    relabelling the loci's read columns by hand.
    """
    cna_tree, snv_dataset, transitions, _ = scored
    perm = np.arange(N_CELLS)[::-1].copy()

    permuted = locus_loglik_batch(
        cna_tree, snv_dataset, transitions, np.arange(snv_dataset.n_snvs),
        alpha=ALPHA, beta=BETA, p0=P0, cell_perm=perm,
    )

    # The same data with the columns already shuffled: tip j carries the reads
    # of cell perm[j].
    tree2, ds2, tr2 = _toy_dataset({
        "clade":     [CELLS[j] for j in range(N_CELLS) if CELLS[perm[j]] in CLADE_CELLS],
        "scattered": [CELLS[j] for j in range(N_CELLS) if CELLS[perm[j]] in SCATTERED_CELLS],
        "sister":    [CELLS[j] for j in range(N_CELLS) if CELLS[perm[j]] in SISTER_CELLS],
        "empty":     [],
    })

    by_hand = locus_loglik_batch(
        tree2, ds2, tr2, np.arange(ds2.n_snvs), alpha=ALPHA, beta=BETA, p0=P0,
    )

    np.testing.assert_allclose(permuted[0], by_hand[0])
    np.testing.assert_allclose(permuted[1], by_hand[1])


def test_permutation_collapses_the_clade_advantage(scored):
    cna_tree, snv_dataset, transitions, observed = scored
    params = _uniform_params(cna_tree)

    observed_gap = (
        observed.loc["clade", "llr_null"] - observed.loc["scattered", "llr_null"]
    )
    assert observed_gap > 20

    gaps = []
    clade_llrs = []
    for seed in range(12):
        df = score_placements(
            cna_tree, snv_dataset, transitions, batch_size=4,
            cell_perm_rng=np.random.default_rng([seed, 1]), **params
        )
        gaps.append(df.loc["clade", "llr_null"] - df.loc["scattered", "llr_null"])
        clade_llrs.append(df.loc["clade", "llr_null"])

    # Shuffling the tips destroys the clade locus's advantage
    assert np.mean(gaps) < 0.25 * observed_gap
    assert np.mean(clade_llrs) < observed.loc["clade", "llr_null"] - 20


def test_permutation_preserves_alt_cell_count_per_locus(scored):
    """The shuffle must move cells between tips, never change their reads."""
    _, snv_dataset, _, _ = scored
    rng = np.random.default_rng(0)
    perm = rng.permutation(snv_dataset.n_leaves)

    ks = snv_dataset.ks
    np.testing.assert_array_equal(
        np.sort(ks[:, perm], axis=1), np.sort(ks, axis=1)
    )
    np.testing.assert_array_equal(ks[:, perm].sum(axis=1), ks.sum(axis=1))


# ── Subsampling ───────────────────────────────────────────────────────────

def test_snv_mask_selects_exactly_those_loci_with_unchanged_values(scored):
    cna_tree, snv_dataset, transitions, full = scored
    params = _uniform_params(cna_tree)

    mask = np.zeros(snv_dataset.n_snvs, dtype=bool)
    mask[[0, 2]] = True
    subset = score_placements(
        cna_tree, snv_dataset, transitions, batch_size=2, snv_mask=mask, **params
    )

    expected_ids = list(snv_dataset.snv_ids[mask])
    assert sorted(subset.index) == sorted(expected_ids)
    pd.testing.assert_frame_equal(
        subset.sort_index(), full.loc[expected_ids].sort_index()
    )


# ── Status handling ───────────────────────────────────────────────────────

def test_cn_zero_clade_yields_non_ok_statuses_with_nan_margin():
    """
    A locus in a segment where most branches are impossible (CN=0) must be
    flagged, not silently scored: llr_margin is NaN when fewer than two
    branches are finite, rather than +inf.
    """
    all_node_names = [n.name for n in _toy_tree()[0].traverse()]
    cn_zero_everywhere = {name: 0 for name in all_node_names}
    cna_tree, snv_dataset, transitions = _toy_dataset(
        {"clade": CLADE_CELLS}, cn_by_node=cn_zero_everywhere
    )
    df = score_placements(
        cna_tree, snv_dataset, transitions, **_uniform_params(cna_tree)
    )

    status = df.loc["clade", "score_status"]
    assert status != SCORE_STATUS_OK
    assert status in (SCORE_STATUS_NO_VALID_BRANCH, SCORE_STATUS_NO_MARGIN)
    assert np.isnan(df.loc["clade", "llr_margin"])
    assert not np.isposinf(df.loc["clade", "llr_margin"])

    # Non-ok loci still receive a MAP placement under the existing logic
    assert df.loc["clade", "node"] in ["Null"] + [
        cna_tree.idx_to_ete[i].name for i in range(cna_tree.n_nodes)
    ]
