import numpy as np

from sntree.constants import MU_ERR
from sntree.likelihood.leaf_emissions import prepare_leaf_emissions
from sntree.likelihood.loglik_all_snvs import map_batch
from sntree.likelihood.locus_loglik_batch import locus_loglik_batch, leaf_inside_weights


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
    print_progress=False,
):
    """Hard EM using shared emissions and batchwise sufficient statistics."""
    alpha, beta = float(init_alpha), float(init_beta)
    N = cna_tree.n_nodes
    placements, history = {}, []
    last_ll = None
    for it in range(max_iter):
        alpha_num = alpha_denom = beta_num = beta_denom = total_ll = 0.0
        placements.clear()
        if print_progress:
            print(f"[EM iter {it}]: alpha={alpha:.4f}, beta={beta:.4f}")

        for snv_idx_list in snv_dataset.snvs_by_seg.values():
            for i0 in range(0, len(snv_idx_list), batch_size):
                batch = np.asarray(snv_idx_list[i0:i0 + batch_size])
                ks = snv_dataset.ks[batch]
                cov = snv_dataset.ns[batch] > 0
                force_null = (ks.sum(axis=1) < min_alt_reads) | ((ks > 0).sum(axis=1) < min_alt_cells)
                emissions = prepare_leaf_emissions(
                    cna_tree, snv_dataset, batch, alpha, beta, p0,
                    p1_fp_mode, responsibilities=True,
                )
                logL_nodes, logL_null = locus_loglik_batch(
                    cna_tree, snv_dataset, transitions, batch, emissions=emissions,
                )
                nodes, scores = map_batch(logL_nodes, logL_null, force_null, pi0)
                finite = np.isfinite(scores)
                total_ll += float(scores[finite].sum())
                for j, snv_idx in enumerate(batch):
                    placements[snv_dataset.snv_ids[snv_idx]] = (
                        int(nodes[j]) if nodes[j] >= 0 else None, float(scores[j])
                    )

                # One-hot branch responsibilities retain null/filtered semantics.
                weights = np.zeros((N, len(batch)))
                placed = np.flatnonzero((nodes >= 0) & finite)
                weights[nodes[placed], placed] = 1.0
                inside = leaf_inside_weights(cna_tree, weights)
                eligible = cov & finite[:, None]
                outside = (1.0 - inside) * eligible
                inside *= eligible
                alpha_num += float(np.sum(outside * emissions.fp_responsibility))
                alpha_denom += float(outside.sum())
                beta_num += float(np.sum(inside * emissions.tp_responsibility))
                beta_denom += float(inside.sum())

        alpha_new = alpha_num / alpha_denom if alpha_denom else alpha
        beta_new = 1.0 - beta_num / beta_denom if beta_denom else beta
        alpha = damp * alpha + (1.0 - damp) * alpha_new
        beta = damp * beta + (1.0 - damp) * beta_new
        history.append(dict(iter=it, alpha=alpha, beta=beta, ll=total_ll))
        if print_progress:
            print(f"    [EM iter {it}]: ll={total_ll:.4f}, alpha_new={alpha_new:.4f}, beta_new={beta_new:.4f}")
        if last_ll is not None:
            if abs(total_ll - last_ll) < tol and max(abs(alpha-alpha_new), abs(beta-beta_new)) < tol:
                break
        last_ll = total_ll
    return alpha, beta, placements, history
