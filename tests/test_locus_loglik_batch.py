"""Placement scores for singleton, unary, multifurcating, and unbalanced trees."""

import numpy as np
import pandas as pd
import pytest
from ete4 import Tree
from scipy.stats import binom

from sntree.io.io_preprocess import build_all
from sntree.likelihood.locus_loglik_batch import locus_loglik_batch


@pytest.mark.parametrize('newick', [
    'a;', '(a)root;', '((a)x,b,c)root;', '(((a,b)x,c)y,d)root;',
])
def test_batch_matches_named_leaf_enumeration(newick):
    tree = Tree(newick, parser=1)
    names = [leaf.name for leaf in tree.leaves()]
    for node in tree.traverse():
        node.props['CN_profile'] = {0: {'cn_tot': 2, 'cn_a': 1, 'cn_b': 1}}
        if node.is_leaf:
            node.props['cell_id'] = node.name
    rng = np.random.default_rng(72)
    ks = rng.integers(0, 11, size=(4, len(names)))
    ns = np.full_like(ks, 10)
    ks[-1], ns[-1] = 0, 0
    ct, ds, transitions = build_all(
        tree, pd.DataFrame({'idx': [0]}), None,
        pd.DataFrame(ns-ks, columns=names), pd.DataFrame(ks, columns=names),
        pd.DataFrame({'snv': np.arange(4), 'cna_idx': 0}),
    )
    actual, null = locus_loglik_batch(ct, ds, transitions, np.arange(4), .1, .2, p1_fp_mode=.3)
    present = np.logaddexp(np.log(.8) + binom.logpmf(ks, ns, .5), np.log(.2) + binom.logpmf(ks, ns, .001))
    absent = np.logaddexp(np.log(.1) + binom.logpmf(ks, ns, .3), np.log(.9) + binom.logpmf(ks, ns, .001))
    for u, node in enumerate(ct.idx_to_ete):
        descendants = {leaf.name for leaf in node.leaves()}
        mask = np.array([name in descendants for name in names])
        expected = np.where(mask, present, absent).sum(axis=1)
        np.testing.assert_allclose(actual[u], expected, atol=1e-12)
    np.testing.assert_allclose(null, absent.sum(axis=1), atol=1e-12)
    np.testing.assert_array_equal(actual[:, -1], 0.)
