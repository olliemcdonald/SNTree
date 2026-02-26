import numpy as np
from sntree.constants import NEG_INF, MU_ERR
from sntree.likelihood.quick_null_stats import quick_null_stats
from sntree.likelihood.locus_loglik_batch import locus_loglik_batch

def loglik_all_snvs(cna_tree,
                    snv_dataset,
                    transitions,
                    alpha=0.0,
                    beta=0.0,
                    p0=MU_ERR,
                    pi0=0.0,
                    p1_fp_mode="one_over_c",
                    min_alt_reads=2,
                    min_alt_cells=2,
                    batch_size=64):
    """
    Fully vectorized SNV placement + log-likelihood evaluator.
    Returns:
        placements : { snv_id : (best_node_idx_or_None, logL) }
        total_ll   : sum of best log-likelihoods
    """

    placements = {}
    total_ll = 0.0

    depth = cna_tree.depth
    N_nodes = cna_tree.n_nodes

    # iterate by CNA segment
    for seg, snv_idx_list in snv_dataset.snvs_by_seg.items():

        if len(snv_idx_list) == 0:
            continue

        # process batches
        for i0 in range(0, len(snv_idx_list), batch_size):
            batch_idx = snv_idx_list[i0 : i0 + batch_size]     # list of SNV indices
            B = len(batch_idx)

            # ---------- 1. Quick null test ----------
            batch_full = []
            batch_null = []
            null_logL = {}

            for snv_idx in batch_idx:
                ll0, total_alt, alt_cells = quick_null_stats(
                    cna_tree, snv_dataset, snv_idx,
                    alpha=alpha, p0=p0, p1_fp_mode=p1_fp_mode
                )
                null_logL[snv_idx] = ll0

                if total_alt < min_alt_reads or alt_cells < min_alt_cells:
                    batch_null.append(snv_idx)
                else:
                    batch_full.append(snv_idx)

            # ---------- 2. Handle trivial null SNVs ----------
            for snv_idx in batch_null:
                snv_id = snv_dataset.snv_ids[snv_idx]
                log_prior_null = np.log(pi0) if pi0 > 0 else NEG_INF
                ll_hat = null_logL[snv_idx] + log_prior_null
                # Skip impossible null placements
                if np.isneginf(ll_hat):
                    placements[snv_id] = (None, ll_hat)
                    continue
                placements[snv_id] = (None, ll_hat)
                total_ll += ll_hat

            if len(batch_full) == 0:
                continue

            # ---------- 3. Vectorized DP for full batch ----------
            # returns:
            #   logL_nodes : shape (N_nodes, B)
            #   logL_null  : shape (B,)
            logL_nodes, logL_null = locus_loglik_batch(
                cna_tree,
                snv_dataset,
                transitions,
                batch_full,
                alpha=alpha,
                beta=beta,
                p0=p0,
                p1_fp_mode=p1_fp_mode,
                include_edge=False
            )

            # ---------- 4. MAP placement for each SNV ----------
            for j, snv_idx in enumerate(batch_full):

                snv_id = snv_dataset.snv_ids[snv_idx]

                # node + null priors
                log_prior_null = np.log(pi0) if pi0 > 0 else NEG_INF
                log_prior_node = np.log((1-pi0) / N_nodes) if pi0 < 1 else NEG_INF

                # scores:
                #   node_score[u] = logL_nodes[u,j] + log_prior_node
                #   null_score    = logL_null[j]    + log_prior_null
                node_scores = logL_nodes[:, j] + log_prior_node
                null_score  = logL_null[j]      + log_prior_null

                # best node (with depth tie-break)
                best_node = None
                best_score = null_score
                best_depth = -1

                # compare nodes
                u_best = np.argmax(node_scores)
                u_score = node_scores[u_best]

                # if node beats null
                if u_score > best_score or (u_score == best_score and depth[u_best] > best_depth):
                    best_node  = u_best
                    best_score = u_score
                    best_depth = depth[u_best]

                # ----- SKIP SNVs whose best likelihood is -inf -----
                if np.isneginf(best_score):
                    # Record placement but DO NOT add to total_ll
                    placements[snv_id] = (best_node, float(best_score))
                    continue
                placements[snv_id] = (best_node, float(best_score))
                total_ll += best_score

    return placements, total_ll
