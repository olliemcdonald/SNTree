"""Regression check that refined SNV assignments name the nodes they were scored on."""

import unittest
from unittest import mock

import pandas as pd
from ete4 import Tree

from sntree.phylo import refinement


class TestRefineGroupPlacements(unittest.TestCase):
    def test_assignments_match_final_tree_after_nni(self):
        cells = ["a", "b", "c", "d"]
        snvs = ["s_c", "s_ab", "s_cd"]
        # s_c is carried by c only, s_ab by a and b, s_cd by c and d.
        alt = pd.DataFrame(
            [[0, 0, 10, 0], [10, 10, 0, 0], [0, 0, 10, 10]],
            index=snvs, columns=cells,
        )
        ref = 20 - alt
        snv_df = pd.DataFrame({"snv": snvs, "cna_idx": [0, 0, 0]})

        mrca = Tree("R;")
        mrca.add_prop("CN_profile", {0: {"cn_tot": 2}})

        with mock.patch.object(
            refinement, "filter_snvs_for_group_from_assignments", return_value=snvs
        ), mock.patch.object(
            refinement, "vcf_subset_to_tables", return_value=(None, ref, alt)
        ), mock.patch.object(
            refinement, "_build_dataset_for_group", return_value=(snv_df, ref, alt)
        ):
            res = refinement.refine_group_tree(
                cells, mrca, {}, None, None, None, "unused.vcf", None,
                alpha=0.01, beta=0.01, p0=0.01,
            )

        tree = Tree(res.newick, parser=1)
        clade = {
            n.name: frozenset(leaf.name for leaf in n.leaves())
            for n in tree.traverse()
        }
        assigned = dict(zip(res.assignments["snv"], res.assignments["node"]))
        self.assertEqual(
            {s: clade[node] for s, node in assigned.items()},
            {
                "s_c": frozenset(["c"]),
                "s_ab": frozenset(["a", "b"]),
                "s_cd": frozenset(["c", "d"]),
            },
        )


if __name__ == "__main__":
    unittest.main()
