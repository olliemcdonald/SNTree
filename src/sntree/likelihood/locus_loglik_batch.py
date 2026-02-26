import numpy as np
from sntree.likelihood.numba_kernels import logpmf_binom, logsumexp2_matrix, logsumexp_axis2
from sntree.constants import NEG_INF, MU_ERR


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
):
    """
    Fully vectorized batched locus DP.
    Returns:
        logL_nodes: (N_nodes, B)
        logL_null:  (B,)
    """

    batch = np.asarray(batch_indices, dtype=int)
    B = batch.shape[0]
    N = cna_tree.n_nodes

    # segment
    seg = int(snv_dataset.seg[batch[0]])

    # extract matrices (B, L)
    ks = snv_dataset.ks[batch]
    ns = snv_dataset.ns[batch]
    L = ks.shape[1]

    # error mixture logs
    log1mb = np.log(1 - beta) if beta < 1.0 else NEG_INF
    logb   = np.log(beta)      if beta > 0.0 else NEG_INF
    loga   = np.log(alpha)     if alpha > 0.0 else NEG_INF
    log1ma = np.log(1 - alpha) if alpha < 1.0 else NEG_INF

    # p1 false positive vector (L,)
    if isinstance(p1_fp_mode, (int, float)):
        p1_vec = np.full(L, float(p1_fp_mode))
    else:
        # p1 = 1/c_leaf
        p1_vec = np.zeros(L)
        for leaf_idx in range(L):
            node_idx = cna_tree.leaf_to_node[leaf_idx]
            c_leaf = cna_tree.CN[node_idx, seg]
            p1_vec[leaf_idx] = (1.0 / c_leaf) if c_leaf > 0 else 0.0

    # Precompute absent-channel leaf likelihood (B, L)
    lp1 = logpmf_binom(ks, ns, p1_vec)
    lp0 = logpmf_binom(ks, ns, p0)

    F_absent_leaf = logsumexp2_matrix(loga+lp1, log1ma + lp0)

    # Initialize DP storage
    # F_present[node]: array shape (B, c_node+1)
    F_present = [None] * N
    F_absent  = np.zeros((N, B))  # internal absent = sum of child absent

    # Leaf mapping
    leaf_order = cna_tree.leaf_order
    node_of_leaf = cna_tree.node_of_leaf

    # ---------- UPWARD DP ----------
    for u in cna_tree.postorder:
        c_u = cna_tree.CN[u, seg]

        if cna_tree.is_leaf[u]:
            # column where this SNV leaf appears
            lf = node_of_leaf[u]
            col = leaf_order[lf]

            # Present-channel likelihood at leaf
            n = ns[:, col]  # (B,)
            k = ks[:, col]  # (B,)

            if c_u == 0:
                lp0  = logpmf_binom(k, n, p0)
                lpA0 = logpmf_binom(k, n, 0.0)
                F_present[u] = logsumexp2_matrix(log1mb + lpA0, logb + lp0)#[:, None] at end previously
            else:
                a = np.arange(c_u + 1) / c_u               # multiplicity fractions
                lpA = logpmf_binom(k[:,None], n[:,None], a)  # (B, c_u+1)

                lp0_scalar = logpmf_binom(k, n, p0)          # (B,)
                lp0_vec    = np.tile(lp0_scalar[:,None], (1, lpA.shape[1]))  # (B, c_u+1)

                F_present[u] = logsumexp2_matrix(log1mb + lpA, logb + lp0_vec)

            # absent-channel at leaf
            F_absent[u] = F_absent_leaf[:, col]

        else:
            # internal node: sum children absent, convolution for present
            Fp = np.zeros((B, c_u+1))
            Fa = np.zeros(B)

            for ch in cna_tree.children[u]:
                Fa += F_absent[ch]

                # convolution
                eid = cna_tree.edge_id[u][ch]
                logM = transitions.logM[eid][seg]    # (c_u+1, c_ch+1)
                Fc = F_present[ch]                   # (B, c_ch+1)

                # vectorized convolution
                Fp += logsumexp_axis2(logM[None,:,:] + Fc[:,None,:])

            F_present[u] = Fp
            F_absent[u] = Fa

    # ---------- DOWNWARD ABSENT DP (G0) ----------
    G0 = np.zeros((N, B))
    root = cna_tree.preorder[0]

    for parent in cna_tree.preorder:
        for child in cna_tree.children[parent]:
            sib_sum = np.sum(
                [F_absent[s] for s in cna_tree.children[parent] if s != child],
                axis=0
            )
            G0[child] = G0[parent] + sib_sum

    # ---------- Compute node likelihoods ----------
    logL_nodes = np.full((N, B), NEG_INF)

    for u in range(N):
        F_u = F_present[u]
        if F_u.shape[1] > 1:
            inside = F_u[:,1]
        else:
            inside = np.full(B, NEG_INF)

        outside = G0[u] if u != root else np.zeros(B)
        logL_nodes[u] = inside + outside

    # edge factor if needed
    if include_edge:
        for u in range(N):
            if u == root: continue
            p = cna_tree.parent[u]
            eid = cna_tree.edge_id[p][u]
            logM_pu = transitions.logM[eid][seg]
            if logM_pu.shape[1] > 1:
                logL_nodes[u] += logM_pu[0,1]
            else:
                logL_nodes[u] += NEG_INF

    # null likelihood
    logL_null = F_absent[root]   # shape (B,)

    return logL_nodes, logL_null