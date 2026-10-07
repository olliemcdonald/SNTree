from sntree.constants import MU_ERR
from sntree.diploid.locus_loglik_batch_diploid import locus_loglik_batch_diploid


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
):
    """Fixed-diploid batched locus likelihood.

    All present descendants are scored at VAF 0.5 and all absent-cell
    one-copy/error components are scored at 0.5.  ``transitions`` and the
    CN-dependent ``p1_fp_mode`` are retained only to preserve the public
    interface used by ML, hard EM, soft EM, and scoring; neither affects this
    diploid model.

    cell_perm : array of leaf-column indices or None
        Optional permutation of the cell → tip assignment.  Column j (the tip
        occupying leaf column j) is scored against cell cell_perm[j]'s read
        counts, leaving every cell's (k, n) pair intact.  Used to build a
        permutation null for clade concordance; None leaves the data untouched.

        The permutation is applied to the read matrices rather than to
        cna_tree.leaf_order because p1_vec below is indexed by leaf column, so
        permuting leaf_order would move the reads without moving the p1 used in
        the absent channel — silently changing the null model.

    Returns:
        logL_nodes: (N_nodes, B)
        logL_null:  (B,)
    """

    return locus_loglik_batch_diploid(
        cna_tree,
        snv_dataset,
        batch_indices,
        alpha=alpha,
        beta=beta,
        p0=p0,
        p1_fp=0.5,
        cell_perm=cell_perm,
    )
