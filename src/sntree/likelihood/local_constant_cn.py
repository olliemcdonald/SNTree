"""
Constant-CN local likelihood for subtree refinement.

Assumptions:
    - All nodes in the subtree share identical total CN.
    - No CNA transitions below MRCA.
    - Expected VAF for present mutation = 1 / CN.
    - No transition matrix, no DP, no multiplicity convolution.

Designed to be used inside NNI or tree rearrangement (SPR) refinement.
"""

from typing import Dict, Tuple
import numpy as np # type: ignore
from ete4 import Tree # type: ignore

from sntree.likelihood.numba_kernels import logpmf_binom, logsumexp2_matrix


# -------------------------------------------------------------------------
# 1. Build per-SNV per-leaf cache (static across NNI)
# -------------------------------------------------------------------------

def build_local_likelihood_cache(
    ks: np.ndarray,              # (S, L)
    ns: np.ndarray,              # (S, L)
    snv_segments: np.ndarray,    # (S,) segment index per SNV
    cn_profile: Dict[int, Dict[str, int]],
    alpha: float,
    beta: float,
    p0: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute delta and total_absent for constant-CN subtree model.

    CN is constant across subtree, but may vary by genomic segment.
    """

    S, L = ks.shape

    # --- Build per-SNV p1 vector ---
    p1_vec = np.zeros(S, dtype=np.float64)

    for s in range(S):
        seg = int(snv_segments[s])
        cn_seg = cn_profile[seg]["cn_tot"]

        if cn_seg <= 0:
            p1_vec[s] = 0.0
        else:
            p1_vec[s] = 1.0 / cn_seg

    # Broadcast to (S, L)
    p1_mat = p1_vec[:, None]

    # Binomial logpmf
    lp_present = logpmf_binom(ks, ns, p1_mat)
    lp_absent  = logpmf_binom(ks, ns, p0)

    log1mb = np.log(1.0 - beta)
    logb   = np.log(beta)
    loga   = np.log(alpha)
    log1ma = np.log(1.0 - alpha)

    # Present-channel mixture
    L_present = logsumexp2_matrix(
        log1mb + lp_present,
        logb   + lp_absent
    )

    # Absent-channel mixture
    L_absent = logsumexp2_matrix(
        loga   + lp_present,
        log1ma + lp_absent
    )

    delta = L_present - L_absent               # (S, L)
    total_absent = np.sum(L_absent, axis=1)    # (S,)

    return delta, total_absent


# -------------------------------------------------------------------------
# 2. Precompute subtree leaf indices (per tree topology)
# -------------------------------------------------------------------------

def compute_descendant_leaf_indices(tree: Tree):
    """
    Returns:
        node_list        : list of nodes in postorder
        node_to_leaves   : dict[node_idx] -> np.array of leaf indices
    """

    node_list = list(tree.traverse("postorder"))
    node_index = {node: i for i, node in enumerate(node_list)}

    leaf_nodes = list(tree.leaves())
    leaf_index = {leaf: i for i, leaf in enumerate(leaf_nodes)}

    node_to_leaves = {}

    for i, node in enumerate(node_list):
        if node.is_leaf:
            node_to_leaves[i] = np.array([leaf_index[node]], dtype=int)
        else:
            idxs = []
            for ch in node.children:
                ch_i = node_index[ch]
                idxs.append(node_to_leaves[ch_i])
            node_to_leaves[i] = np.concatenate(idxs)

    return node_list, node_to_leaves


# -------------------------------------------------------------------------
# 3. Scoring using cached likelihood
# -------------------------------------------------------------------------

def score_tree_local_cached(
    tree: Tree,
    delta: np.ndarray,          # (S, L)
    total_absent: np.ndarray,   # (S,)
) -> Tuple[Dict[int, int], float]:
    """
    Score tree using precomputed delta.

    Returns:
        placements : dict {snv_idx: best_node_idx_or_None}
        total_ll   : float
    """

    node_list, node_to_leaves = compute_descendant_leaf_indices(tree)

    S, L = delta.shape
    N = len(node_list)

    placements = {}
    total_ll = 0.0

    for s in range(S):

        best_score = total_absent[s]
        best_node = None

        for u in range(N):
            desc = node_to_leaves[u]

            # subtree sum
            score_u = total_absent[s] + np.sum(delta[s, desc])

            if score_u > best_score:
                best_score = score_u
                best_node = u

        placements[s] = best_node
        total_ll += best_score

    return placements, total_ll


# -------------------------------------------------------------------------
# 4. Score wrapper
# -------------------------------------------------------------------------

def make_local_cached_scorer(
    delta: np.ndarray,
    total_absent: np.ndarray,
):
    """
    Returns a function score(tree) -> (placements, total_ll)
    """

    def score(tree: Tree):
        return score_tree_local_cached(tree, delta, total_absent)

    return score