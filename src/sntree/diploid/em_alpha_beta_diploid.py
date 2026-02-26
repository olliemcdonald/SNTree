import numpy as np

from sntree.likelihood.numba_kernels import logpmf_binom
from sntree.constants import NEG_INF, MU_ERR
from .quick_null_stats_diploid import quick_null_stats_diploid
from .locus_loglik_batch_diploid import locus_loglik_batch_diploid


def sigmoid_logdiff(log1, log0):
    return 1.0 / (1.0 + np.exp(log0 - log1))


def em_alpha_beta_diploid(
    cna_tree,
    snv_dataset,
    init_alpha=0.0,
    init_beta=0.0,
    p0=MU_ERR,
    pi0=0.0,
    max_iter=25,
    tol=1e-4,
    damp=0.5,
    min_alt_reads=2,
    min_alt_cells=2,
    batch_size=64,
    print_progress=False
):

    alpha = float(init_alpha)
    beta  = float(init_beta)

    depth = cna_tree.depth
    leaf_desc = cna_tree.leaf_desc_mask
    leaf_order = cna_tree.leaf_order
    N = cna_tree.n_nodes
    L = snv_dataset.n_leaves

    history = []
    last_ll = None

    for it in range(max_iter):
        if print_progress:
            print(f"[EM diploid iter {it}] alpha={alpha:.4f}, beta={beta:.4f}")

        q_out = []   # responsibilities for false positives
        w_in  = []   # responsibilities for dropout
        total_ll = 0.0

        for seg, snv_idx_list in snv_dataset.snvs_by_seg.items():

            for i0 in range(0, len(snv_idx_list), batch_size):
                batch = snv_idx_list[i0:i0+batch_size]

                batch_null = {}
                batch_full = []

                # quick null test
                for snv_idx in batch:
                    ll0, tot_alt, alt_cells = quick_null_stats_diploid(
                        cna_tree, snv_dataset, snv_idx, p0=p0
                    )
                    if tot_alt < min_alt_reads or alt_cells < min_alt_cells:
                        batch_null[snv_idx] = ll0
                    else:
                        batch_full.append(snv_idx)

                # process nulls
                for snv_idx, ll0 in batch_null.items():
                    total_ll += ll0 + (np.log(pi0) if pi0 > 0 else NEG_INF)

                    ks = snv_dataset.ks[snv_idx]
                    ns = snv_dataset.ns[snv_idx]

                    mask_cov = ns > 0
                    k = ks[mask_cov]
                    n = ns[mask_cov]

                    # responsibilities for false positives
                    log1 = np.log(max(alpha,1e-12))    + logpmf_binom(k,n,0.5)
                    log0 = np.log(max(1-alpha,1e-12))  + logpmf_binom(k,n,p0)
                    q_out.extend(sigmoid_logdiff(log1, log0).tolist())

                if not batch_full:
                    continue

                # DP
                logL_nodes, logL_null = locus_loglik_batch_diploid(
                    cna_tree, snv_dataset, batch_full, alpha=alpha, beta=beta, p0=p0
                )

                log_prior_node = np.log((1-pi0)/N) if pi0 < 1 else NEG_INF
                log_prior_null = np.log(pi0) if pi0 > 0 else NEG_INF

                # MAP + responsibilities
                for j, snv_idx in enumerate(batch_full):

                    ks = snv_dataset.ks[snv_idx]
                    ns = snv_dataset.ns[snv_idx]

                    # MAP decision
                    node_scores = logL_nodes[:, j] + log_prior_node
                    null_score  = logL_null[j]      + log_prior_null

                    best_node = None
                    best_score = null_score
                    best_depth = -1

                    u_best = np.argmax(node_scores)
                    if node_scores[u_best] > best_score or \
                       (node_scores[u_best] == best_score and depth[u_best] > best_depth):
                        best_node = u_best
                        best_score = node_scores[u_best]
                        best_depth = depth[u_best]

                    total_ll += best_score

                    # responsibilities
                    if best_node is None:
                        leaf_inside = np.zeros(L, dtype=bool)
                    else:
                        leaf_inside = leaf_desc[best_node]
                    leaf_outside = ~leaf_inside

                    mask_cov = ns > 0

                    # outside (G=0)
                    mask = leaf_outside & mask_cov
                    k = ks[mask]
                    n = ns[mask]
                    log1 = np.log(max(alpha,1e-12))    + logpmf_binom(k,n,0.5)
                    log0 = np.log(max(1-alpha,1e-12))  + logpmf_binom(k,n,p0)
                    q_out.extend(sigmoid_logdiff(log1, log0).tolist())

                    # inside (G=1)
                    mask = leaf_inside & mask_cov
                    k = ks[mask]
                    n = ns[mask]

                    # mixture responsibilities for dropout
                    # compare:
                    #   1: mutated -> Binom(0.5)
                    #   0: dropout -> Binom(p0)
                    log1 = np.log(max(1-beta,1e-12)) + logpmf_binom(k,n,0.5)
                    log0 = np.log(max(beta,1e-12))   + logpmf_binom(k,n,p0)
                    w_in.extend(sigmoid_logdiff(log1, log0).tolist())

        # M-step
        alpha_new = np.mean(q_out) if q_out else alpha
        beta_new  = 1.0 - (np.mean(w_in) if w_in else beta)

        alpha = damp*alpha + (1-damp)*alpha_new
        beta  = damp*beta  + (1-damp)*beta_new

        history.append(dict(iter=it, alpha=alpha, beta=beta, ll=total_ll))

        if last_ll is not None:
            if abs(total_ll - last_ll) < tol and \
               max(abs(alpha-alpha_new), abs(beta-beta_new)) < tol:
                break

        last_ll = total_ll

    return alpha, beta, history