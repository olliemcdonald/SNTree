# sntree/workflow/preprocess_tree.py

import os
import time
from ete4 import Tree  # type: ignore
from sntree.io.io_tree import read_raw_medicc_tree


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def clean_identical_clades(
    tree: Tree,
    cna_distance_tsv: str,
) -> Tree:
    """
    Collapse internal structure within CNA-identical clades.

    For each zero-distance CNA group:
        - Identify its MRCA
        - Identify child subtrees fully contained in the group
        - Remove internal structure inside those subtrees
        - Reattach leaves directly under MRCA
        - Leave cousin branches untouched
    """

    from sntree.phylo.refinement import (
        load_cna_distance_matrix,
        cna_zero_groups,
        map_groups_to_mrca,
    )

    print("[preprocess] Collapsing CN-identical clades")

    dist_df = load_cna_distance_matrix(cna_distance_tsv, drop_diploid=True)
    groups = cna_zero_groups(dist_df)
    groups_mrca = map_groups_to_mrca(tree, groups)

    n_collapsed = 0

    for group_cells, mrca in groups_mrca:

        if len(group_cells) <= 1:
            continue

        group_set = set(group_cells)

        # Identify child subtrees of MRCA that are fully within the group
        group_children = []

        for child in mrca.children:

            descendant_leaves = {
                node.name
                for node in child.traverse()
                if node.is_leaf
            }

            if descendant_leaves.issubset(group_set):
                group_children.append(child)

        if len(group_children) <= 1:
            continue

        print(
            f"[preprocess] Collapsing MRCA {mrca.name} "
            f"(n_subtrees={len(group_children)})"
        )

        # Collect all leaves inside group subtrees
        leaves_to_reattach = []
        for child in group_children:
            leaves_to_reattach.extend(
                [n for n in child.traverse() if n.is_leaf]
            )

        # Detach only the group subtrees
        for child in group_children:
            child.detach()

        # Reattach leaves directly under MRCA
        for leaf in leaves_to_reattach:
            leaf.detach()
            mrca.add_child(leaf)

        n_collapsed += 1

    print(f"[preprocess] Total clades collapsed: {n_collapsed}")

    return tree

def run_preprocess(sample, output_root, input_paths):

    print(f"[{now()}] Preprocess stage started for sample {sample}")
    t0 = time.time()

    sample_out = os.path.join(output_root, sample, "sntree")
    os.makedirs(sample_out, exist_ok=True)

    # ---- Load and normalize MEDICC2 tree ----
    print(f"[{now()}] Loading raw MEDICC2 tree...")
    tree = read_raw_medicc_tree(input_paths.medicc_tree)

    # ---- Clean CN-identical cousin structure ----
    print(f"[{now()}] Cleaning CN-identical clades...")
    tree = clean_identical_clades(tree, input_paths.cna_distances)

    # Ensure root naming persists
    tree.name = "root"

    # ---- Write preprocessed tree ----
    output_path = input_paths.preprocessed_tree
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(tree.write(parser=1).strip() + "\n")

    runtime = time.time() - t0

    print(f"[{now()}] Preprocess complete.")
    print(f"[{now()}] Output written to: {output_path}")
    print(f"[{now()}] Runtime: {runtime:.2f} sec")
