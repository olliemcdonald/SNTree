# quick_null_stats.py

import numpy as np
from sntree.likelihood.numba_kernels import logpmf_binom, logsumexp2_matrix
from sntree.constants import NEG_INF, MU_ERR

def quick_null_stats(cna_tree, snv_dataset, snv_idx,
                     alpha=0.0, p0=MU_ERR, p1_fp_mode="one_over_c"):
    """
    Fast approximate null log-likelihood and basic counts, under G = 0 everywhere.
    
    Array-based version of quick_null_stats:
    - cna_tree: CNATree
    - snv_dataset: SNVDataset
    - snv_idx: integer index of SNV in dataset
    """

    seg = snv_dataset.seg[snv_idx]
    ks = snv_dataset.ks[snv_idx]   # shape (N_leaves,)
    ns = snv_dataset.ns[snv_idx]   # shape (N_leaves,)

    # Count total alt reads and alt-cells
    total_alt = int(ks.sum())
    alt_cells = int((ks > 0).sum())

    # Nothing to compute if no coverage anywhere
    if ns.sum() == 0:
        return 0.0, total_alt, alt_cells

    # alpha mixture: G=0 channel at leaves
    if isinstance(p1_fp_mode, (int, float)):
        p1_scalar = float(p1_fp_mode)
    else:
        # p1 = 1/c_leaf; need c_leaf from tree.CN
        # c_leaf_idx maps SNV leaf-order to tree leaf-order
        # but cna_tree.leaf_order maps tree leaf_idx -> dataset leaf_idx
        c_leaf_vec = np.zeros_like(ns, dtype=float)
        for leaf_node_idx in range(cna_tree.n_nodes):
            leaf_idx = cna_tree.node_of_leaf[leaf_node_idx]
            if leaf_idx != -1:
                dataset_leaf_col = cna_tree.leaf_order[leaf_idx]
                c_leaf_vec[dataset_leaf_col] = cna_tree.CN[leaf_node_idx, seg]
        with np.errstate(divide="ignore", invalid="ignore"):
            p1_vec = np.where(c_leaf_vec > 0, 1.0 / c_leaf_vec, 0.0)
    # If scalar, broadcast
    if isinstance(p1_fp_mode, (int, float)):
        p1_vec = np.full_like(ns, p1_scalar, dtype=float)

    # Precompute logs
    loga   = np.log(alpha)        if alpha > 0.0 else NEG_INF
    log1ma = np.log(1 - alpha)    if alpha < 1.0 else NEG_INF

    # Compute logL per leaf
    valid = ns > 0
    ks_v = ks[valid]
    ns_v = ns[valid]
    p1_v = p1_vec[valid]

    lp1 = logpmf_binom(ks_v, ns_v, p1_v)
    lp0 = logpmf_binom(ks_v, ns_v, p0)

    leaf_null_ll = logsumexp2_matrix(loga + lp1, log1ma + lp0)

    logL_null = float(leaf_null_ll.sum())

    return logL_null, total_alt, alt_cells