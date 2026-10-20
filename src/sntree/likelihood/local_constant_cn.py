"""Fixed-diploid local likelihood for CNA-identical subtree refinement.

Present mutations use VAF 0.5. CN profiles affect only the configurable
false-positive emission; no CNA transitions are evaluated.
"""

from typing import Dict, Mapping, Sequence, Tuple
import numpy as np # type: ignore
from ete4 import Tree # type: ignore

from sntree.likelihood.leaf_emissions import leaf_emissions, validate_fp_mode


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
    p1_fp_mode="one_over_c",
) -> Tuple[np.ndarray, np.ndarray]:
    """Cache leaf contributions with a fixed present VAF and separate FP VAF."""
    mode = validate_fp_mode(p1_fp_mode)
    if isinstance(mode, float):
        p1_fp = mode
    else:
        cn = np.array([cn_profile[int(seg)]["cn_tot"] for seg in snv_segments])
        p1_fp = np.divide(1.0, cn, out=np.zeros(cn.shape, dtype=float), where=cn > 0)[:, None]
    emissions = leaf_emissions(ks, ns, p1_fp, alpha, beta, p0)
    delta = emissions.present - emissions.absent
    total_absent = emissions.absent.sum(axis=1)

    return delta, total_absent


# -------------------------------------------------------------------------
# 2. Precompute subtree leaf indices (per tree topology)
# -------------------------------------------------------------------------

def compute_descendant_leaf_indices(tree: Tree, leaf_columns: Mapping[str, int]):
    """
    Resolve descendants to fixed cache columns by cell identity, not traversal
    order, which changes during tree rearrangements.

    Returns:
        node_list        : list of nodes in postorder
        node_to_leaves   : dict[node_idx] -> np.array of leaf indices
    """

    node_list = list(tree.traverse("postorder"))
    node_index = {node: i for i, node in enumerate(node_list)}

    node_to_leaves = {}

    for i, node in enumerate(node_list):
        if node.is_leaf:
            cell_id = node.props.get("cell_id", node.name)
            node_to_leaves[i] = np.array([leaf_columns[cell_id]], dtype=int)
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
    leaf_columns: Mapping[str, int],
) -> Tuple[Dict[int, int], float]:
    """
    Score tree using precomputed delta.

    Returns:
        placements : dict {snv_idx: best_node_idx_or_None}
        total_ll   : float
    """

    node_list, node_to_leaves = compute_descendant_leaf_indices(tree, leaf_columns)

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
    leaf_names: Sequence[str],
):
    """
    Returns a function score(tree) -> (placements, total_ll).

    leaf_names must give the cell identities in delta's column order.
    """
    leaf_columns = {name: i for i, name in enumerate(leaf_names)}
    if len(leaf_names) != delta.shape[1] or len(leaf_columns) != len(leaf_names):
        raise ValueError("leaf_names must uniquely identify every cache column")

    def score(tree: Tree):
        return score_tree_local_cached(tree, delta, total_absent, leaf_columns)

    return score
