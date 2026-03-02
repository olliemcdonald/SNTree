import os
import time
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd # type: ignore
import numpy as np # type: ignore
from ete4 import Tree # type: ignore

from sntree.io.io_cna import add_cna, barcode_cell_maps, cna_lookups
from sntree.io.io_snv import vcf_subset_to_tables, _build_dataset_for_group
from sntree.io.snv_index import SNVIndex, filter_snvs_for_group_from_assignments
from sntree.io.io_preprocess import build_snv_dataset
from sntree.phylo.tree_search import (
  _assign_internal_names,
  _presence_distance,
  neighbor_joining,
  NNI_step
)
from sntree.phylo.tree_utils import (
    _set_leaf_cell_ids,
    _attach_constant_cn,
    _tree_to_newick
)
from sntree.likelihood.local_constant_cn import (
    build_local_likelihood_cache,
    make_local_cached_scorer,
)

@dataclass
class GroupResult:
    """
    Class for group of identical cells by CNA
    """
    group_id: str
    cells: List[str]
    newick: str
    total_ll: float
    assignments: pd.DataFrame

def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def load_cna_distance_matrix(path: str, drop_diploid: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", index_col=0)
    if drop_diploid:
        if "diploid" in df.index:
            df = df.drop(index="diploid")
        if "diploid" in df.columns:
            df = df.drop(columns="diploid")
    return df


def cna_zero_groups(dist_df: pd.DataFrame) -> List[List[str]]:
    """
    Single-linkage groups on distance == 0.0.
    """
    samples = list(dist_df.index)
    idx = {s: i for i, s in enumerate(samples)}
    n = len(samples)
    adj = [set() for _ in range(n)]

    for i in range(n):
        row = dist_df.iloc[i].values
        zero_idx = np.where(row == 0.0)[0]
        for j in zero_idx:
            if i == j:
                continue
            adj[i].add(j)
            adj[j].add(i)

    visited = [False] * n
    groups = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(samples[u])
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        groups.append(sorted(comp))
    return groups


def map_groups_to_mrca(tree: Tree, groups: List[List[str]]) -> List[Tuple[List[str], Tree]]:
    out = []
    for g in groups:
        if len(g) == 1:
            out.append((g, tree[g[0]]))
            continue
        mrca = tree.common_ancestor(g)
        out.append((g, mrca))
    return out


def _placements_to_assignments(ete_tree: Tree, dataset, placements: Dict[int, Optional[int]],) -> pd.DataFrame:
    node_list = list(ete_tree.traverse("postorder"))
    rows = []
    for snv_idx, node_idx in placements.items():
        snv_id = dataset.snv_ids[snv_idx]

        if node_idx is None:
            node_name = "null"
        else:
            node_name = node_list[node_idx].name

        rows.append((snv_id, node_name))

    return pd.DataFrame(rows, columns=["snv", "node"])

def refine_group_tree(
    group_cells: Sequence[str],
    mrca_node: Tree,
    snv_assignments: Dict[str, str],
    cna_profiles: pd.DataFrame,
    sample_mapping: pd.DataFrame,
    cna_idx: pd.DataFrame,
    vcf_path: str,
    snv_index: SNVIndex,
    normal_name: Optional[str] = None,
    name_map: Optional[Dict[str, str]] = None,
    exclude_outside: bool = False,
    alpha: float = 0.0,
    beta: float = 0.0,
    p0: float = 0.0,
    batch_size: int = 64,
    max_iters: int = 50,
    init: str = "nj",
) -> Optional[GroupResult]:

    group_cells = list(group_cells)
    if name_map:
        group_cells = [name_map.get(c, c) for c in group_cells]
    if len(group_cells) == 0:
        return None

    # Filter SNVs using the prior assignments
    snv_ids = filter_snvs_for_group_from_assignments(
        snv_index,
        snv_assignments,
        mrca_node,
        group_cells,
        exclude_root=True,
        exclude_null=True,
    )

    if len(snv_ids) == 0:
        return None

    snv_ids_set = set(snv_ids)
    _, ref_df, alt_df = vcf_subset_to_tables(
        vcf_path,
        snv_ids_set,
        normal_name=normal_name,
        name_map=name_map,
        cells_keep=set(group_cells),
    )
    if ref_df.shape[0] == 0:
        return None

    snv_df, ref_df, alt_df = _build_dataset_for_group(ref_df, alt_df, cna_idx)
    if ref_df.shape[0] == 0:
        return None

    # ---- Initialize topology ----

    if init == "nj":
        pres = (alt_df.to_numpy(dtype=int) > 0)
        dist = _presence_distance(pres)
        ete_tree = neighbor_joining(dist, list(ref_df.columns))
    else:
        leaves = ",".join(ref_df.columns)
        ete_tree = Tree(f"({leaves});")

    _assign_internal_names(ete_tree, root_name=mrca_node.name)
    _set_leaf_cell_ids(ete_tree)

    # ---- Constant CN from MRCA ----

    cn_profile = mrca_node.props["CN_profile"]
    _attach_constant_cn(ete_tree, cn_profile)

    # extract constant CN value (assumes identical across segments)
    first_seg = next(iter(cn_profile.values()))
    cn_value = first_seg["cn_tot"]

    # ---- Build dataset matrices ----

    dataset = build_snv_dataset(ref_df, alt_df, snv_df, list(ref_df.columns))

    ks = dataset.ks
    ns = dataset.ns

    # ---- Precompute local likelihood cache ----

    delta, total_absent = build_local_likelihood_cache(
        ks,
        ns,
        dataset.seg,        # segment per SNV
        cn_profile,
        alpha,
        beta,
        p0,
    )

    score = make_local_cached_scorer(delta, total_absent)

    # ---- Initial scoring ----

    placements, best_ll = score(ete_tree)

    # ---- Hill climb via NNI ----

    improved = True
    it = 0

    while improved and it < max_iters:
        improved = False
        it += 1

        new_tree, new_ll, new_pl, improved = NNI_step(
            ete_tree,
            best_ll,
            score,
        )

        if improved:
            ete_tree = new_tree
            best_ll = new_ll
            placements = new_pl

    # ---- Convert placements to assignment table ----

    # Map local node indices back to node names
    assignments = _placements_to_assignments(
        ete_tree,
        dataset,
        placements,
    )

    return GroupResult(
        group_id="",
        cells=group_cells,
        newick=_tree_to_newick(ete_tree),
        total_ll=best_ll,
        assignments=assignments,
    )


def refine_all_groups(
    tree: Tree,
    snv_assignments: Dict[str, str],
    cna_profiles: pd.DataFrame,
    sample_mapping: pd.DataFrame,
    cna_distance_tsv: str,
    vcf_path: str,
    snv_index: SNVIndex,
    output_dir: str,
    normal_name: Optional[str] = None,
    name_map: Optional[Dict[str, str]] = None,
    exclude_outside: bool = False,
    alpha: float = 0.0,
    beta: float = 0.0,
    p0: float = 0.0,
    batch_size: int = 64,
    max_iters: int = 50,
    init: str = "nj",
) -> List[GroupResult]:

    print(f"[{now()}] Starting subtree refinement")
    t0_total = time.time()

    os.makedirs(output_dir, exist_ok=True)

    if "CN_profile" not in tree.props:
        tree = add_cna(tree, sample_mapping, cna_profiles)

    cna_idx, _ = cna_lookups(cna_profiles)

    # Build barcode maps
    barcode_to_cell, cell_to_barcode = barcode_cell_maps(sample_mapping)

    dist_df = load_cna_distance_matrix(cna_distance_tsv, drop_diploid=True)
    groups = cna_zero_groups(dist_df)
    groups_mrca = map_groups_to_mrca(tree, groups)

    final_assignments = dict(snv_assignments)

    n_groups = sum(1 for g, _ in groups_mrca if len(g) > 1)
    print(f"[{now()}] {n_groups} CNA-identical groups with >1 cell found")

    results: List[GroupResult] = []

    for i, (group_cells, mrca) in enumerate(groups_mrca):

        if len(group_cells) <= 1:
            continue

        group_id = f"group_{i:03d}"
        newick_path = os.path.join(output_dir, f"{group_id}.newick")
        assign_path = os.path.join(output_dir, f"{group_id}_snv_assignments.tsv")

        # ----------------------------------------------------
        # Resume support: restore subtree if already exists
        # ----------------------------------------------------
        if os.path.exists(newick_path) and os.path.exists(assign_path):

            print(f"[{now()}] Restoring completed group {i+1} (cells={len(group_cells)})")

            refined_subtree = Tree(open(newick_path).read().strip(), parser=1)

            # Convert subtree leaves to BARCODE
            for leaf in refined_subtree.leaves():
                leaf.name = cell_to_barcode.get(leaf.name, leaf.name)

            # Graft subtree
            # Detach current children under MRCA
            for child in list(mrca.children):
                child.detach()
            # Attach saved subtree children
            for child in list(refined_subtree.children):
                child.detach()
                mrca.add_child(child)

            # Merge SNV assignments
            group_assign_df = pd.read_csv(assign_path, sep="\t")
            for _, row in group_assign_df.iterrows():
                final_assignments[row["snv"]] = row["node"]

            continue

        # ----------------------------------------------------
        # Otherwise run refinement
        # ----------------------------------------------------
        print(
            f"[{now()}] Refining group {i+1}/{len(groups_mrca)} "
            f"(cells={len(group_cells)}, mrca={mrca.name})"
        )
        t_group = time.time()

        res = refine_group_tree(
            group_cells,
            mrca,
            snv_assignments,
            cna_profiles,
            sample_mapping,
            cna_idx,
            vcf_path,
            snv_index,
            normal_name=normal_name,
            name_map=barcode_to_cell,
            exclude_outside=exclude_outside,
            alpha=alpha,
            beta=beta,
            p0=p0,
            batch_size=batch_size,
            max_iters=max_iters,
            init=init,
        )

        if res is None:
            continue

        # Load subtree
        refined_subtree = Tree(res.newick.strip(), parser=1)

        # Convert CELL names back to BARCODE names
        for leaf in refined_subtree.leaves():
            if leaf.name in cell_to_barcode:
                leaf.name = cell_to_barcode[leaf.name]
            else:
                warnings.warn(
                    f"[refine] Leaf name '{leaf.name}' not found in cell_to_barcode map. "
                    "Leaving as-is."
                )

        # Update res.newick (BARCODE version)
        res.newick = refined_subtree.write(parser=1).strip()

        # Convert assignment nodes to BARCODE
        res.assignments["node"] = res.assignments["node"].apply(
            lambda x: cell_to_barcode.get(x, x)
        )

        # Graft subtree into main tree
        for child in list(mrca.children):
            child.detach()

        for child in list(refined_subtree.children):
            child.detach()
            mrca.add_child(child)

        # Update global SNV assignments
        for _, row in res.assignments.iterrows():
            final_assignments[row["snv"]] = row["node"]

        res.group_id = group_id
        results.append(res)

        # Write outputs (BARCODE)
        with open(newick_path, "w") as f:
            f.write(res.newick + "\n")

        res.assignments.to_csv(assign_path, sep="\t", index=False)

        with open(os.path.join(output_dir, f"{group_id}_likelihood.txt"), "w") as f:
            f.write(f"{res.total_ll}\n")

        group_runtime = time.time() - t_group
        print(
            f"[{now()}] Finished group {i+1} "
            f"(runtime={group_runtime:.2f} sec, "
            f"loglik={res.total_ll:.4f})"
        )

    total_runtime = time.time() - t0_total
    print(f"[{now()}] All groups refined.")
    print(f"[{now()}] Total refinement runtime: {total_runtime:.2f} sec")

    return results, final_assignments