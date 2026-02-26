import numpy as np
from sntree.likelihood.numba_kernels import logpmf_binom
from sntree.constants import NEG_INF, MU_ERR
from sntree.likelihood.quick_null_stats import quick_null_stats
from sntree.likelihood.locus_loglik_batch import locus_loglik_batch


def sigmoid_logdiff(log1, log0):
    return 1.0 / (1.0 + np.exp(log0 - log1))


def em_alpha_beta(
    cna_tree,
    snv_dataset,
    transitions,
    init_alpha=0.0,
    init_beta=0.0,
    p0=MU_ERR,
    pi0=0.0,
    p1_fp_mode="one_over_c",
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

    L = snv_dataset.n_leaves
    N = cna_tree.n_nodes
    depth = cna_tree.depth
    leaf_desc = cna_tree.leaf_desc_mask
    leaf_order = cna_tree.leaf_order
    leaf_to_node = cna_tree.leaf_to_node

    placements = {}
    history = []
    last_ll = None

    for it in range(max_iter):

        q_out = []
        w_in  = []
        total_ll = 0.0
        placements.clear()

        if print_progress:
            print(f"[EM iter {it}]: alpha={alpha:.4f}, beta={beta:.4f}")

        # Iterate by segment
        for seg, snv_idx_list in snv_dataset.snvs_by_seg.items():

            for i0 in range(0, len(snv_idx_list), batch_size):
                batch = snv_idx_list[i0:i0+batch_size]

                # ---------- quick null test ----------
                batch_null = []
                batch_full = []
                null_logL = {}

                for snv_idx in batch:
                    ll0, tot_alt, alt_cells = quick_null_stats(
                        cna_tree, snv_dataset, snv_idx,
                        alpha=alpha, p0=p0, p1_fp_mode=p1_fp_mode
                    )
                    null_logL[snv_idx] = ll0

                    if tot_alt < min_alt_reads or alt_cells < min_alt_cells:
                        batch_null.append(snv_idx)
                    else:
                        batch_full.append(snv_idx)

                # ---------- process trivial nulls ----------
                for snv_idx in batch_null:
                    snv_id = snv_dataset.snv_ids[snv_idx]
                    ll = null_logL[snv_idx] + (np.log(pi0) if pi0>0 else NEG_INF)
                    # Skip impossible null placements
                    if np.isneginf(ll):
                        placements[snv_id] = (None, ll)
                        continue
                    placements[snv_id] = (None, ll)
                    total_ll += ll

                    # responsibilities (vectorized)
                    ks = snv_dataset.ks[snv_idx]
                    ns = snv_dataset.ns[snv_idx]

                    # p1 fp vector for whole leaves
                    if isinstance(p1_fp_mode, (int, float)):
                        p1 = np.full(L, float(p1_fp_mode))
                    else:
                        # p1 = 1/c_leaf
                        p1 = np.zeros(L)
                        for leaf_idx in range(L):
                            node_idx = leaf_to_node[leaf_idx]
                            c_leaf = cna_tree.CN[node_idx, seg]
                            p1[leaf_idx] = (1.0/c_leaf) if c_leaf>0 else 0.0

                    # leaves with coverage
                    mask_cov = ns > 0
                    k = ks[mask_cov]
                    n = ns[mask_cov]
                    p1v = p1[mask_cov]

                    log1_vec = np.log(max(alpha,1e-12))    + logpmf_binom(k,n,p1v)
                    log0_vec = np.log(max(1-alpha,1e-12))  + logpmf_binom(k,n,p0)
                    qs = sigmoid_logdiff(log1_vec, log0_vec)

                    q_out.extend(qs.tolist())

                # ---------- DP for full batch ----------
                if not batch_full:
                    continue

                # vectorized DP
                logL_nodes, logL_null = locus_loglik_batch(
                    cna_tree, snv_dataset, transitions,
                    batch_full,
                    alpha=alpha,
                    beta=beta,
                    p0=p0,
                    p1_fp_mode=p1_fp_mode,
                    include_edge=False,
                )

                # ---------- MAP + responsibilities ----------
                for j, snv_idx in enumerate(batch_full):
                    snv_id = snv_dataset.snv_ids[snv_idx]

                    # priors
                    B = logL_null.shape[0]  # number of SNVs in batch_full
                    n_nodes_scored = N
                    log_node_pr = np.log((1-pi0) / n_nodes_scored) if pi0<1 else NEG_INF
                    log_null_pr = np.log(pi0) if pi0>0 else NEG_INF

                    # MAP over all nodes and null
                    node_scores = logL_nodes[:, j] + log_node_pr
                    null_score  = logL_null[j] + log_null_pr

                    # find best
                    best_node = None
                    best_val  = null_score
                    best_depth = -1

                    if null_score < np.max(node_scores):
                        # some node beats null
                        u = np.argmax(node_scores)
                        best_node = u
                        best_val  = node_scores[u]
                        best_depth = depth[u]

                    # Skip SNVs whose MAP likelihood is -inf
                    if np.isneginf(best_val):
                        placements[snv_id] = (best_node, float(best_val))
                        continue

                    placements[snv_id] = (best_node, float(best_val))
                    total_ll += best_val

                    # responsibilities
                    ks = snv_dataset.ks[snv_idx]
                    ns = snv_dataset.ns[snv_idx]

                    # leaf masks
                    if best_node is None:
                        leaf_inside = np.zeros(L, dtype=bool)
                    else:
                        leaf_inside = leaf_desc[best_node]
                    leaf_outside = ~leaf_inside

                    # p1 
                    if isinstance(p1_fp_mode, (int, float)):
                        p1 = np.full(L, float(p1_fp_mode))
                    else:
                        p1 = np.zeros(L)
                        for leaf_idx in range(L):
                            node_idx = leaf_to_node[leaf_idx]
                            c_leaf = cna_tree.CN[node_idx, seg]
                            p1[leaf_idx] = (1.0/c_leaf) if c_leaf>0 else 0.0

                    # coverage mask
                    mask_cov = ns > 0

                    # outside responsibilities
                    mask = leaf_outside & mask_cov
                    k = ks[mask]
                    n = ns[mask]
                    p1v = p1[mask]

                    log1 = np.log(max(alpha,1e-12))    + logpmf_binom(k,n,p1v)
                    log0 = np.log(max(1-alpha,1e-12)) + logpmf_binom(k,n,p0)
                    q_out.extend(sigmoid_logdiff(log1, log0).tolist())

                    # inside responsibilities
                    if isinstance(p1_fp_mode, (int,float)):
                        p_eff = np.full(L, 0.0)
                    else:
                        p_eff = np.zeros(L)
                        for leaf_idx in range(L):
                            node_idx = leaf_to_node[leaf_idx]
                            c_leaf = cna_tree.CN[node_idx, seg]
                            p_eff[leaf_idx] = (1.0/c_leaf) if c_leaf>0 else 0.0

                    mask = leaf_inside & mask_cov
                    k = ks[mask]
                    n = ns[mask]
                    pAv = p_eff[mask]

                    log1 = np.log(max(1-beta,1e-12)) + logpmf_binom(k,n,pAv)
                    log0 = np.log(max(beta,1e-12))   + logpmf_binom(k,n,p0)
                    w_in.extend(sigmoid_logdiff(log1, log0).tolist())

        # ---------- M-STEP ----------
        alpha_new = np.mean(q_out) if q_out else alpha
        beta_new  = 1.0 - np.mean(w_in) if w_in else beta

        alpha = damp*alpha + (1-damp)*alpha_new
        beta  = damp*beta  + (1-damp)*beta_new

        history.append(dict(iter=it, alpha=alpha, beta=beta, ll=total_ll))

        if print_progress:
            print(f"    [EM iter {it}]: ll={total_ll:.4f}, alpha_new={alpha_new:.4f}, beta_new={beta_new:.4f}")

        if last_ll is not None:
            if abs(total_ll - last_ll) < tol and max(abs(alpha-alpha_new), abs(beta-beta_new)) < tol:
                break

        last_ll = total_ll

    return alpha, beta, placements, history