import numpy as np
#from scipy.stats import binom
from sntree.likelihood.numba_kernels import logpmf_binom
from sntree.constants import NEG_INF, MU_ERR

def locus_loglik_batch_diploid(
    cna_tree,
    snv_dataset,
    batch_indices,
    alpha=0.0,
    beta=0.0,
    p0=MU_ERR,
    p1_fp=0.5,   # false positive library genotype rate
):
    """
    Diploid likelihood with correct alpha/beta formulation:
      Inside:  (1-beta)*Binom(n,0.5) + beta*Binom(n,p0)
      Outside: (1-alpha)*Binom(n,p0) + alpha*Binom(n,p1_fp)
    """

    batch = np.asarray(batch_indices, dtype=int)
    ks = snv_dataset.ks[batch]      # (B, L)
    ns = snv_dataset.ns[batch]      # (B, L)
    B, L = ks.shape

    N = cna_tree.n_nodes
    leaf_desc = cna_tree.leaf_desc_mask

    mask_cov = ns > 0

    # Binomial components
    ll_half = np.full((B, L), 0.0)
    #ll_half[mask_cov] = binom.logpmf(ks[mask_cov], ns[mask_cov], 0.5)
    ll_half[mask_cov] = logpmf_binom(ks[mask_cov], ns[mask_cov], 0.5)

    ll_p0 = np.full((B, L), 0.0)
    #ll_p0[mask_cov] = binom.logpmf(ks[mask_cov], ns[mask_cov], p0)
    ll_p0[mask_cov] = logpmf_binom(ks[mask_cov], ns[mask_cov], p0)

    ll_p1 = np.full((B, L), 0.0)
    #ll_p1[mask_cov] = binom.logpmf(ks[mask_cov], ns[mask_cov], p1_fp)
    ll_p1[mask_cov] = logpmf_binom(ks[mask_cov], ns[mask_cov], p1_fp)

    # Inside mixture (G = 1)
    inside_logL = np.logaddexp(
        np.log(max(1 - beta, 1e-12)) + ll_half,
        np.log(max(beta,      1e-12)) + ll_p0
    )

    # Outside mixture (G = 0)
    outside_logL = np.logaddexp(
        np.log(max(1 - alpha, 1e-12)) + ll_p0,
        np.log(max(alpha,      1e-12)) + ll_p1
    )

    # Node likelihoods
    logL_nodes = np.zeros((N, B))
    for u in range(N):
        inside_mask  = leaf_desc[u]
        outside_mask = ~inside_mask

        inside_sum  = np.sum(inside_logL[:, inside_mask], axis=1)
        outside_sum = np.sum(outside_logL[:, outside_mask], axis=1)

        logL_nodes[u] = inside_sum + outside_sum

    # Null likelihood: absent everywhere → outside everywhere
    logL_null = np.sum(outside_logL, axis=1)

    return logL_nodes, logL_null