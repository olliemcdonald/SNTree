import numpy as np

from sntree.constants import NEG_INF, MU_ERR
from sntree.likelihood.leaf_emissions import prepare_leaf_emissions
from sntree.likelihood.locus_loglik_batch import locus_loglik_batch


def map_batch(logL_nodes, logL_null, force_null, pi0, prefer_branch_ties=False):
    """MAP placements with the existing uniform node/null priors and filtering."""
    N, B = logL_nodes.shape
    log_node_prior = np.log((1.0 - pi0) / N) if pi0 < 1 else NEG_INF
    log_null_prior = np.log(pi0) if pi0 > 0 else NEG_INF
    nodes = np.argmax(logL_nodes, axis=0)
    node_scores = logL_nodes[nodes, np.arange(B)] + log_node_prior
    null_scores = logL_null + log_null_prior
    take_node = node_scores >= null_scores if prefer_branch_ties else node_scores > null_scores
    take_node &= ~force_null
    return np.where(take_node, nodes, -1), np.where(take_node, node_scores, null_scores)


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
    """Fixed-diploid MAP placements and the sum of finite best log likelihoods."""
    placements = {}
    total_ll = 0.0
    for snv_idx_list in snv_dataset.snvs_by_seg.values():
        for i0 in range(0, len(snv_idx_list), batch_size):
            batch = np.asarray(snv_idx_list[i0:i0 + batch_size])
            ks = snv_dataset.ks[batch]
            force_null = (ks.sum(axis=1) < min_alt_reads) | ((ks > 0).sum(axis=1) < min_alt_cells)
            emissions = prepare_leaf_emissions(
                cna_tree, snv_dataset, batch, alpha, beta, p0, p1_fp_mode,
            )
            logL_nodes, logL_null = locus_loglik_batch(
                cna_tree, snv_dataset, transitions, batch, emissions=emissions,
            )
            nodes, scores = map_batch(logL_nodes, logL_null, force_null, pi0, True)
            total_ll += float(scores[np.isfinite(scores)].sum())
            for j, snv_idx in enumerate(batch):
                placements[snv_dataset.snv_ids[snv_idx]] = (
                    int(nodes[j]) if nodes[j] >= 0 else None, float(scores[j])
                )
    return placements, total_ll
