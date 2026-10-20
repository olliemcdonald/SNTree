"""Batched fixed-diploid placement likelihoods in the current SNTree backend."""

from sntree.constants import MU_ERR
from sntree.likelihood.leaf_emissions import prepare_leaf_emissions
from sntree.likelihood.numba_kernels import (
    placement_loglik_postorder,
    branch_weights_to_leaves,
)


def locus_loglik_batch(
    cna_tree,
    snv_dataset,
    transitions,
    batch_indices,
    alpha=0.0,
    beta=0.0,
    p0=MU_ERR,
    p1_fp_mode="one_over_c",
    include_edge=False,
    cell_perm=None,
    *,
    emissions=None,
):
    """Return node (N, B) and null (B,) log likelihoods.

    Present mutations have VAF 0.5; only the false-positive emission can
    depend on CN. ``transitions`` is retained for caller compatibility and
    may be None. CNA edge factors are not part of this model.

    ``cell_perm`` permutes read columns among fixed tips, preserving each
    cell's (k, n) pair and each tip's false-positive emission.
    Internal EM callers can reuse ``emissions`` for their M-step.
    """
    if include_edge:
        raise ValueError("CNA edge factors are not supported by the fixed-diploid model")
    if emissions is None:
        emissions = prepare_leaf_emissions(
            cna_tree, snv_dataset, batch_indices, alpha, beta, p0,
            p1_fp_mode, cell_perm,
        )
    return placement_loglik_postorder(
        emissions.present, emissions.absent,
        cna_tree.parent, cna_tree.postorder,
        cna_tree.node_of_leaf, cna_tree.leaf_order,
    )


def leaf_inside_weights(cna_tree, branch_weights):
    """Accumulate branch responsibilities without an N-by-L descendant mask."""
    return branch_weights_to_leaves(
        branch_weights, cna_tree.parent, cna_tree.preorder,
        cna_tree.node_of_leaf, cna_tree.leaf_order,
    )
