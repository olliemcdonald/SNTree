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

    ks = snv_dataset.ks[snv_idx]   # shape (N_leaves,)
    ns = snv_dataset.ns[snv_idx]   # shape (N_leaves,)

    # Count total alt reads and alt-cells
    total_alt = int(ks.sum())
    alt_cells = int((ks > 0).sum())

    # Nothing to compute if no coverage anywhere
    if ns.sum() == 0:
        return 0.0, total_alt, alt_cells

    # Fixed-diploid model: CN values and p1_fp_mode are deliberately ignored.
    p1_vec = np.full_like(ns, 0.5, dtype=float)

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
