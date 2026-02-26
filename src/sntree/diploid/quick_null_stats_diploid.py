import numpy as np
#from scipy.stats import binom
from sntree.likelihood.numba_kernels import logpmf_binom
from sntree.constants import NEG_INF, MU_ERR

def quick_null_stats_diploid(cna_tree, snv_dataset, snv_idx, p0=MU_ERR, alpha=0.0):
    """
    Diploid null likelihood: SNV absent everywhere.
    L = Π_leaves Binom(k, n, p0)
    No copy-number effects.
    Returns:
        logL_null, total_alt, alt_cells
    """
    ks = snv_dataset.ks[snv_idx]   # (L,)
    ns = snv_dataset.ns[snv_idx]   # (L,)

    mask = ns > 0
    if not np.any(mask):
        # no coverage anywhere → likelihood = 0
        return 0.0, 0, 0

    k = ks[mask]
    n = ns[mask]

    #logL = np.sum(binom.logpmf(k, n, p0))
    logL = np.sum(logpmf_binom(k, n, p0))
    total_alt = int(np.sum(k))
    alt_cells = int(np.sum(k > 0))

    return float(logL), total_alt, alt_cells