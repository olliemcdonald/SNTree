"""Regression checks for stable cell identities during cached NNI scoring."""

import unittest

import numpy as np
from ete4 import Tree

from sntree.likelihood.local_constant_cn import make_local_cached_scorer
from sntree.phylo.tree_search import NNI_step


class TestCachedCellIndexing(unittest.TestCase):
    def setUp(self):
        self.cells = ["a", "b", "c", "d"]
        # One locus supported by a and b, plus a locus best assigned to null.
        self.delta = np.array([[9., 8., -10., -10.], [-1., -2., -3., -4.]])
        self.absent = np.array([-30., -5.])
        self.score = make_local_cached_scorer(
            self.delta, self.absent, self.cells
        )

    def named_placements(self, tree, placements):
        nodes = list(tree.traverse("postorder"))
        return {
            s: None if u is None else frozenset(
                leaf.props.get("cell_id", leaf.name) for leaf in nodes[u].leaves()
            )
            for s, u in placements.items()
        }

    def test_sibling_order_and_tree_copy_preserve_score_and_placement(self):
        for newick in ["(((a,b),c),d);", "((c,(b,a)),d);"]:
            tree = Tree(newick)
            for candidate in [tree, tree.copy()]:
                placements, ll = self.score(candidate)
                self.assertEqual(ll, -18.)
                self.assertEqual(
                    self.named_placements(candidate, placements),
                    {0: frozenset(["a", "b"]), 1: None},
                )

    def test_cache_column_order_can_differ_from_tree_order(self):
        perm = [2, 0, 3, 1]
        score = make_local_cached_scorer(
            self.delta[:, perm], self.absent, [self.cells[i] for i in perm]
        )
        tree = Tree("(((a,b),c),d);")
        self.assertEqual(score(tree), self.score(tree))

    def test_different_named_topology_changes_score(self):
        placements, ll = self.score(Tree("((a,c),(b,d));"))
        self.assertEqual(ll, -26.)
        self.assertEqual(
            self.named_placements(Tree("((a,c),(b,d));"), placements),
            {0: frozenset(["a"]), 1: None},
        )

    def test_cell_id_is_used_when_leaf_labels_differ(self):
        tree = Tree("((tip_a,tip_b),(tip_c,tip_d));")
        for leaf in tree.leaves():
            leaf.props["cell_id"] = leaf.name.removeprefix("tip_")
        placements, ll = self.score(tree.copy())
        self.assertEqual(ll, -18.)
        self.assertEqual(
            self.named_placements(tree, placements),
            {0: frozenset(["a", "b"]), 1: None},
        )

    def test_nni_candidates_match_named_reference(self):
        tree = Tree("((a,c),(b,d));")
        initial_clades = {
            frozenset(leaf.name for leaf in node.leaves())
            for node in tree.traverse()
        }
        evaluated = []

        def checked_score(candidate):
            # Independent reference: look up each cell's contribution by name.
            expected = 0.
            for s, baseline in enumerate(self.absent):
                contributions = dict(zip(self.cells, self.delta[s]))
                expected += max(
                    [baseline] + [
                        baseline + sum(contributions[leaf.name] for leaf in node.leaves())
                        for node in candidate.traverse()
                    ]
                )
            placements, ll = self.score(candidate)
            self.assertEqual(ll, expected)
            evaluated.append(ll)
            return placements, ll

        best_tree, best_ll, placements, improved = NNI_step(
            tree, self.score(tree)[1], checked_score
        )
        self.assertTrue(evaluated)
        self.assertTrue(improved)
        self.assertEqual(best_ll, -18.)
        self.assertEqual(self.score(best_tree), (placements, best_ll))
        self.assertEqual(
            initial_clades,
            {frozenset(leaf.name for leaf in node.leaves()) for node in tree.traverse()},
        )
        self.assertEqual(self.score(tree)[1], -26.)
