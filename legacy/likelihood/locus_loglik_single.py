import numpy as np
from scipy.stats import binom
from scipy.special import logsumexp
from sntree.constants import NEG_INF, MU_ERR

def locus_loglik_single(cna_tree,
                        snv_dataset,
                        transitions,
                        snv_idx,
                        alpha=0.0,
                        beta=0.0,
                        p0=MU_ERR,
                        p1_fp_mode="one_over_c",
                        include_edge=False):
    """
    New array-based single-SNV locus likelihood.
    Returns: dict { node_idx : logL, None : logL_null }
    """

    # segment index for this SNV
    seg = int(snv_dataset.seg[snv_idx])

    # leaf-level data (vectors)
    ks = snv_dataset.ks[snv_idx]   # (L,)
    ns = snv_dataset.ns[snv_idx]   # (L,)
    L = len(ks)

    N = cna_tree.n_nodes
    leaf_order = cna_tree.leaf_order

    # -----------------------------
    # Prep error mixture parameters
    # -----------------------------
    log1mb = np.log(1 - beta) if beta < 1.0 else NEG_INF
    logb   = np.log(beta)      if beta > 0.0 else NEG_INF
    loga   = np.log(alpha)     if alpha > 0.0 else NEG_INF
    log1ma = np.log(1 - alpha) if alpha < 1.0 else NEG_INF

    # -----------------------------
    # p1 false-positive model
    # -----------------------------
    if isinstance(p1_fp_mode, (int, float)):
        # constant p1
        p1_vec = np.full(L, float(p1_fp_mode))
    else:
        # p1 = 1/c_leaf
        p1_vec = np.zeros(L, dtype=float)
        for node in range(N):
            lf = cna_tree.node_of_leaf[node]
            if lf != -1:
                col = leaf_order[lf]
                c_leaf = cna_tree.CN[node, seg]
                p1_vec[col] = (1.0 / c_leaf) if c_leaf > 0 else 0.0

    # -----------------------------
    # Allocate DP arrays
    # -----------------------------
    F_present = [None] * N   # each entry is shape (c_node+1,)
    F_absent  = [None] * N   # shape scalar

    # node index → dataset-column index (if leaf)
    node_to_col = np.full(N, -1, dtype=int)
    for node in range(N):
        lf = cna_tree.node_of_leaf[node]
        if lf != -1:
            node_to_col[node] = leaf_order[lf]

    # -----------------------------
    # Helpers for leaf likelihood
    # -----------------------------
    def present_leaf(c_leaf, k, n):
        if c_leaf == 0:
            if n == 0:
                return np.zeros(1)
            lp0  = binom.logpmf(k, n, p0)
            lpA0 = binom.logpmf(k, n, 0.0)
            return np.array([logsumexp([log1mb + lpA0, logb + lp0])])

        if n == 0:
            return np.zeros(c_leaf+1)

        a = np.arange(c_leaf+1) / c_leaf
        lpA = binom.logpmf(k, n, a)
        lp0 = binom.logpmf(k, n, p0)
        lp0_vec = np.full_like(lpA, lp0)
        return logsumexp(
            np.vstack([log1mb + lpA, logb + lp0_vec]),
            axis=0
        )

    def absent_leaf(c_leaf, k, n):
        if n == 0:
            return 0.0
        lp1 = binom.logpmf(k, n, p1_vec)
        lp0 = binom.logpmf(k, n, p0)
        return logsumexp([loga + lp1, log1ma + lp0])

    # -----------------------------
    # UPWARD DP (present + absent)
    # -----------------------------
    for node in cna_tree.postorder:
        c_node = cna_tree.CN[node, seg]

        if cna_tree.is_leaf[node]:
            col = node_to_col[node]
            if col == -1:
                # this leaf has no data
                F_present[node] = np.zeros(c_node+1)
                F_absent[node]  = 0.0
            else:
                k = ks[col]
                n = ns[col]
                F_present[node] = present_leaf(c_node, k, n)
                # absent-channel scalar
                p1 = p1_vec[col]
                if n == 0:
                    F_absent[node] = 0.0
                else:
                    lp1 = binom.logpmf(k, n, p1)
                    lp0 = binom.logpmf(k, n, p0)
                    F_absent[node] = logsumexp([loga + lp1, log1ma + lp0])

        else:
            children = cna_tree.children[node]

            # absent = sum of children
            Fa = 0.0
            # present = sum over children of convolution
            Fp = np.zeros(c_node+1)

            for ch in children:
                Fa += F_absent[ch]

                # convolution with logM
                eid = cna_tree.edge_id[node][ch]
                logM = transitions.logM[eid][seg]  # shape (c_node+1, c_child+1)
                Fch = F_present[ch]

                tmp = logsumexp(logM + Fch[np.newaxis,:], axis=1)
                Fp += tmp

            F_absent[node]  = Fa
            F_present[node] = Fp

    # -----------------------------
    # DOWNWARD DP (absent-only G0)
    # -----------------------------
    G0 = np.zeros(N)
    root = cna_tree.preorder[0]

    for parent in cna_tree.preorder:
        for child in cna_tree.children[parent]:
            # sum absent from siblings
            sib_sum = 0.0
            for sib in cna_tree.children[parent]:
                if sib != child:
                    sib_sum += F_absent[sib]
            G0[child] = G0[parent] + sib_sum

    # -----------------------------
    # Build output: per-node log-likelihood
    # -----------------------------
    out = {}

    for node in range(N):

        F_u = F_present[node]
        if len(F_u) <= 1:
            inside = NEG_INF
        else:
            inside = F_u[1]

        outside = G0[node] if node != root else 0.0

        val = inside + outside

        # optional edge factor
        if include_edge and node != root:
            p = cna_tree.parent[node]
            eid = cna_tree.edge_id[p][node]
            logM_pu = transitions.logM[eid][seg]
            if logM_pu.shape[1] > 1:
                val += logM_pu[0,1]
            else:
                val += NEG_INF

        out[node] = float(val)

    # null placement
    out[None] = float(F_absent[root])

    return out