import numpy as np
from sntree.constants import NEG_INF, MU_ERR

from .quick_null_stats_diploid import quick_null_stats_diploid
from .locus_loglik_batch_diploid import locus_loglik_batch_diploid


def loglik_all_snvs_diploid(
    cna_tree,
    snv_dataset,
    alpha=0.0,
    beta=0.0,
    p0=MU_ERR,
    pi0=0.0,
    min_alt_reads=2,
    min_alt_cells=2,
    batch_size=64
):

    placements = {}
    total_ll = 0.0

    depth = cna_tree.depth
    N_nodes = cna_tree.n_nodes

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

            # null SNVs
            for snv_idx, ll0 in batch_null.items():
                snv_id = snv_dataset.snv_ids[snv_idx]
                ll_hat = ll0 + (np.log(pi0) if pi0 > 0 else NEG_INF)
                placements[snv_id] = (None, ll_hat)
                total_ll += ll_hat

            if not batch_full:
                continue

            # diploid DP
            logL_nodes, logL_null = locus_loglik_batch_diploid(
                cna_tree,
                snv_dataset,
                batch_full,
                alpha=alpha,
                beta=beta,
                p0=p0
            )

            log_prior_null = np.log(pi0) if pi0 > 0 else NEG_INF
            log_prior_node = np.log((1 - pi0) / N_nodes) if pi0 < 1 else NEG_INF

            for j, snv_idx in enumerate(batch_full):

                snv_id = snv_dataset.snv_ids[snv_idx]
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

                placements[snv_id] = (best_node, float(best_score))
                total_ll += best_score

    return placements, total_ll