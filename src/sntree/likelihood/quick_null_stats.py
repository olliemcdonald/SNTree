import numpy as np

from sntree.constants import MU_ERR
from sntree.likelihood.leaf_emissions import prepare_leaf_emissions


def quick_null_stats(cna_tree, snv_dataset, snv_idx,
                     alpha=0.0, p0=MU_ERR, p1_fp_mode="one_over_c"):
    """Exact null likelihood and alternate-read support for one locus."""
    ks = snv_dataset.ks[snv_idx]
    emissions = prepare_leaf_emissions(
        cna_tree, snv_dataset, [snv_idx], alpha=alpha, p0=p0,
        p1_fp_mode=p1_fp_mode,
    )
    return float(emissions.absent.sum()), int(ks.sum()), int(np.count_nonzero(ks > 0))
