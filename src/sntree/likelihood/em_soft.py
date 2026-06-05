import numpy as np

from sntree.likelihood.numba_kernels import logpmf_binom
from sntree.constants import NEG_INF, MU_ERR, EPS
from sntree.likelihood.locus_loglik_batch import locus_loglik_batch


def _sigmoid_logdiff(log1, log0):
    return 1.0 / (1.0 + np.exp(log0 - log1))


def _logsumexp_rows(log_mat):
    """Numerically stable logsumexp over rows (axis=0) of a 2D array."""
    m = log_mat.max(axis=0)
    out = m + np.log(np.exp(log_mat - m[None, :]).sum(axis=0))
    return out


def em_soft(
    cna_tree,
    snv_dataset,
    transitions,
    init_alpha,
    init_beta,
    init_pi0=0.5,
    alpha_dir=1.0,
    p0=MU_ERR,
    p1_fp_mode="one_over_c",
    max_iter=50,
    tol=1e-4,
    joint=False,
    batch_size=1024,
    print_progress=False,
):
    """
    Soft EM for branch proportions pi_b.

    Extends the hard SNTree EM (em_alpha_beta) by keeping branch assignments
    soft throughout.  Every candidate site contributes to every branch with
    weight proportional to its inside-outside likelihood score, recovering
    relative mutation burdens pi_b even for sub-threshold loci.

    The existing locus_loglik_batch DP scores are reused directly as mixture
    components; no new DP computation is required.

    Parameters
    ----------
    cna_tree, snv_dataset, transitions : SNTree data structures
    init_alpha, init_beta : float
        Starting error parameters (from hard EM or user-supplied).
    init_pi0 : float
        Initial null-component weight.
    alpha_dir : float
        Symmetric Dirichlet concentration on pi_b.  1.0 = uniform prior;
        values < 1 encourage sparsity (requires alpha_dir > 0).
    p0 : float
        Background sequencing error probability.
    p1_fp_mode : str or float
        False-positive signal probability per leaf; "one_over_c" uses 1/CN.
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence threshold on max |Δpi_b|.
    joint : bool
        If True, also update alpha and beta with soft-weighted M-step
        (Eqs 9-10 of sntree_extended.tex).  If False, alpha/beta are fixed
        at init_alpha/init_beta (two-phase estimation).
    batch_size : int
        Loci processed per call to locus_loglik_batch.
    print_progress : bool

    Returns
    -------
    alpha : float
    beta : float
    pi_b : np.ndarray, shape (N_nodes,)
        Normalised branch mutation proportions.
    pi_0 : float
        Null-component weight.
    history : list of dict
    """
    alpha = float(init_alpha)
    beta  = float(init_beta)
    pi_0  = float(init_pi0)

    N = cna_tree.n_nodes
    L = snv_dataset.n_leaves
    M = snv_dataset.n_snvs

    # Uniform initialisation of branch proportions
    pi_b = np.full(N, 1.0 / N)

    # leaf_desc_mask (N, L) bool → float for matmul
    leaf_desc = cna_tree.leaf_desc_mask.astype(np.float64)  # (N, L)

    history = []
    last_pi = None

    for iteration in range(max_iter):

        # ── Accumulators ──────────────────────────────────────────────────
        soft_counts = np.zeros(N)   # Σ_m γ_{mb}  for each branch b
        soft_null   = 0.0           # Σ_m γ_{m0}
        total_ll    = 0.0

        # For joint alpha/beta M-step (only populated when joint=True)
        alpha_num   = 0.0
        alpha_denom = 0.0
        beta_num    = 0.0
        beta_denom  = 0.0

        # Log-space parameters (recomputed each iteration)
        log_pi_b  = np.log(np.maximum(pi_b, EPS))          # (N,)
        log_pi_0  = np.log(max(pi_0,       EPS))
        log_1mpi0 = np.log(max(1.0 - pi_0, EPS))
        log_alpha = np.log(max(alpha, EPS))
        log_1ma   = np.log(max(1.0 - alpha, EPS))
        log_beta  = np.log(max(beta,  EPS))
        log_1mb   = np.log(max(1.0 - beta,  EPS))

        if print_progress:
            print(f"[soft EM iter {iteration}]: "
                  f"alpha={alpha:.5f}  beta={beta:.5f}  pi_0={pi_0:.4f}")

        # ── Segment loop (mirrors em_alpha_beta structure) ────────────────
        for seg, snv_idx_list in snv_dataset.snvs_by_seg.items():

            # Per-leaf p1 vector for this segment
            if isinstance(p1_fp_mode, (int, float)):
                p1_vec = np.full(L, float(p1_fp_mode))
            else:  # "one_over_c"
                p1_vec = np.zeros(L)
                for lf in range(L):
                    node_idx = cna_tree.leaf_to_node[lf]
                    c = cna_tree.CN[node_idx, seg]
                    p1_vec[lf] = (1.0 / c) if c > 0 else 0.0

            for i0 in range(0, len(snv_idx_list), batch_size):
                batch = snv_idx_list[i0 : i0 + batch_size]
                B_b   = len(batch)

                # ── Inside-outside DP ─────────────────────────────────────
                # logL_nodes : (N, B_b)  log L_m(b | α, β)
                # logL_null  : (B_b,)    log L_m(∅ | α)
                logL_nodes, logL_null = locus_loglik_batch(
                    cna_tree, snv_dataset, transitions, batch,
                    alpha=alpha, beta=beta, p0=p0,
                    p1_fp_mode=p1_fp_mode,
                )

                # ── E-step ────────────────────────────────────────────────
                # Log unnorm. branch responsibilities: (N, B_b)
                log_u_b = log_1mpi0 + log_pi_b[:, None] + logL_nodes

                # Log unnorm. null responsibility: (B_b,)
                log_u_0 = log_pi_0 + logL_null

                # Stack null + branches → (N+1, B_b), then logsumexp over axis 0
                log_stack = np.vstack([log_u_0[None, :], log_u_b])  # (N+1, B_b)
                log_Z     = _logsumexp_rows(log_stack)               # (B_b,)

                total_ll += log_Z.sum()

                # Normalised responsibilities
                gamma_b = np.exp(log_u_b - log_Z[None, :])  # (N, B_b)
                gamma_0 = np.exp(log_u_0  - log_Z)          # (B_b,)

                # Accumulate soft counts
                soft_counts += gamma_b.sum(axis=1)           # (N,)
                soft_null   += float(gamma_0.sum())

                # ── Joint alpha / beta M-step contributions ───────────────
                if joint:
                    ks  = snv_dataset.ks[np.array(batch)]    # (B_b, L)
                    ns  = snv_dataset.ns[np.array(batch)]    # (B_b, L)
                    cov = ns > 0                             # (B_b, L)

                    # q_leaf : P(FP | k, n, outside cell) — (B_b, L)
                    # Uses the outside-cell (absent-channel) parameterisation
                    lq1 = log_alpha + logpmf_binom(ks, ns, p1_vec)
                    lq0 = log_1ma   + logpmf_binom(ks, ns, p0)
                    q_leaf = _sigmoid_logdiff(lq1, lq0)      # (B_b, L)

                    # w_leaf : P(TP | k, n, inside cell) — (B_b, L)
                    lw1 = log_1mb  + logpmf_binom(ks, ns, p1_vec)
                    lw0 = log_beta + logpmf_binom(ks, ns, p0)
                    w_leaf = _sigmoid_logdiff(lw1, lw0)      # (B_b, L)

                    # inside_weight[m, j] = Σ_b γ_{mb} · 1[j ∈ desc(b)]
                    # gamma_b.T : (B_b, N)  ×  leaf_desc : (N, L)  →  (B_b, L)
                    inside_w = gamma_b.T @ leaf_desc         # (B_b, L)

                    alpha_num   += float(np.sum((1.0 - inside_w) * q_leaf * cov))
                    alpha_denom += float(np.sum((1.0 - inside_w) * cov))
                    beta_num    += float(np.sum(inside_w * w_leaf * cov))
                    beta_denom  += float(np.sum(inside_w * cov))

        # ── M-step: branch proportions ────────────────────────────────────
        # MAP update with symmetric Dirichlet(alpha_dir) prior:
        #   π_b  ∝  (α_dir - 1) + Σ_m γ_{mb}
        # Denominator equals N*(α_dir-1) + Σ_m Σ_b γ_{mb}
        #                   = N*(α_dir-1) + soft_counts.sum()
        branch_denom = N * (alpha_dir - 1.0) + soft_counts.sum()
        if branch_denom > 0:
            pi_b_new = (alpha_dir - 1.0 + soft_counts) / branch_denom
        else:
            pi_b_new = soft_counts / max(soft_counts.sum(), EPS)

        # Safety renormalisation (numerically should already sum to 1)
        s = pi_b_new.sum()
        pi_b_new = pi_b_new / s if s > 0 else np.full(N, 1.0 / N)

        # ── M-step: null weight ───────────────────────────────────────────
        pi_0_new = float(soft_null) / M

        # ── M-step: alpha / beta (only when joint=True) ───────────────────
        if joint:
            if alpha_denom > 0:
                alpha = float(np.clip(alpha_num / alpha_denom, EPS, 1.0 - EPS))
            if beta_denom > 0:
                beta  = float(np.clip(1.0 - beta_num / beta_denom, EPS, 1.0 - EPS))

        history.append(dict(
            iter=iteration,
            alpha=alpha,
            beta=beta,
            pi_0=pi_0_new,
            ll=total_ll,
        ))

        if print_progress:
            top_b = int(np.argmax(pi_b_new))
            print(f"    ll={total_ll:.4f}  pi_0={pi_0_new:.4f}  "
                  f"pi_max={pi_b_new[top_b]:.4f} (node {top_b})")

        # ── Convergence check ─────────────────────────────────────────────
        if last_pi is not None:
            delta = float(np.max(np.abs(pi_b_new - last_pi)))
            if delta < tol:
                pi_b = pi_b_new
                pi_0 = pi_0_new
                break

        last_pi = pi_b_new.copy()
        pi_b    = pi_b_new
        pi_0    = pi_0_new

    return alpha, beta, pi_b, pi_0, history
