"""Shared fixed-diploid emissions and conditional error responsibilities."""

from dataclasses import dataclass
from numbers import Real

import numpy as np

from sntree.constants import MU_ERR
from sntree.likelihood.numba_kernels import logpmf_binom


def validate_fp_mode(mode):
    """Keep the false-positive allele fraction separate from alpha and VAF=0.5."""
    if isinstance(mode, str) and mode == "one_over_c":
        return mode
    if isinstance(mode, Real) and np.isfinite(mode) and 0.0 <= mode <= 1.0:
        return float(mode)
    raise ValueError("p1_fp_mode must be 'one_over_c' or a probability in [0, 1]")


def false_positive_vaf(cna_tree, segment, mode="one_over_c"):
    """False-positive VAF in dataset-column order, independent of tree traversal."""
    mode = validate_fp_mode(mode)
    if isinstance(mode, float):
        return mode
    cn = cna_tree.CN[cna_tree.leaf_to_node, segment]
    vaf = np.zeros(cna_tree.n_leaves, dtype=np.float64)
    vaf[cna_tree.leaf_order] = np.divide(
        1.0, cn, out=np.zeros(cn.shape, dtype=np.float64), where=cn > 0
    )
    return vaf


@dataclass
class LeafEmissions:
    present: np.ndarray
    absent: np.ndarray
    fp_responsibility: np.ndarray = None
    tp_responsibility: np.ndarray = None


def leaf_emissions(ks, ns, p1_fp, alpha, beta, p0=MU_ERR, responsibilities=False):
    """Compute each binomial component once and reuse it for the E/M steps.

    Present: (1-beta) Binom(n, 0.5) + beta Binom(n, p0).
    Absent:  alpha Binom(n, p1_fp) + (1-alpha) Binom(n, p0).
    No-coverage cells contribute exactly zero log likelihood.
    """
    for name, value in (("alpha", alpha), ("beta", beta), ("p0", p0)):
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be a probability in [0, 1]")

    lp0 = logpmf_binom(ks, ns, p0)
    lp_half = lp0 if p0 == 0.5 else logpmf_binom(ks, ns, 0.5)
    if np.all(np.asarray(p1_fp) == 0.5):
        lp_fp = lp_half
    elif np.all(np.asarray(p1_fp) == p0):
        lp_fp = lp0
    else:
        lp_fp = logpmf_binom(ks, ns, p1_fp)

    with np.errstate(divide="ignore"):
        log_fp = np.log(alpha) + lp_fp
        log_tp = np.log1p(-beta) + lp_half
        absent = np.logaddexp(log_fp, np.log1p(-alpha) + lp0)
        present = np.logaddexp(log_tp, np.log(beta) + lp0)
    uncovered = ns == 0
    absent[uncovered] = 0.0
    present[uncovered] = 0.0

    def conditional(log_component, log_mixture):
        # Impossible channels have zero posterior weight, not NaN.
        log_ratio = np.full(log_mixture.shape, -np.inf)
        np.subtract(log_component, log_mixture, out=log_ratio,
                    where=np.isfinite(log_mixture))
        result = np.exp(log_ratio)
        result[uncovered] = 0.0
        return result

    return LeafEmissions(
        present, absent,
        conditional(log_fp, absent) if responsibilities else None,
        conditional(log_tp, present) if responsibilities else None,
    )


def prepare_leaf_emissions(
    cna_tree, snv_dataset, batch_indices, alpha=0.0, beta=0.0, p0=MU_ERR,
    p1_fp_mode="one_over_c", cell_perm=None, responsibilities=False,
):
    batch = np.asarray(batch_indices, dtype=int)
    ks, ns = snv_dataset.ks[batch], snv_dataset.ns[batch]
    if cell_perm is not None:
        ks, ns = ks[:, cell_perm], ns[:, cell_perm]
    segments = snv_dataset.seg[batch]
    if len(batch) and np.any(segments != segments[0]):
        raise ValueError("A likelihood batch must contain a single CNA segment")
    p1_fp = false_positive_vaf(cna_tree, int(segments[0]), p1_fp_mode) if len(batch) else 0.5
    return leaf_emissions(ks, ns, p1_fp, alpha, beta, p0, responsibilities)
