import numpy as np
from collections import defaultdict
from sntree.constants import NEG_INF, MU_ERR


# Creates the multiplicity distribution given a gain as a matrix
def gain_matrix(c: int) -> np.ndarray:
    M = np.zeros((c+1, c+2))
    a = np.arange(c+1)
    M[np.arange(c+1), a]   += 1 - a/c
    M[np.arange(c+1), a+1] += a/c
    return M


# Creates the multiplicity distribution given a loss as a matrix
def loss_matrix(c: int) -> np.ndarray:
    M = np.zeros((c+1, c))
    if c == 0: return M
    rows = np.arange(1, c+1)      # a>=1 -> a-1
    M[rows, rows-1] += rows / c
    rows = np.arange(0, c)        # a<=c-1 -> stay
    M[rows, rows]   += 1 - rows / c
    return M


# Compose exactly m unit steps; cache per (c_start, delta)
_multi_cache = {}
# This just chains multiple gains/losses in the same function
# It works like stepping through +1's or -1's delta times
def multi_step_matrix(c_start: int, delta: int):
    key = (c_start, delta)
    hit = _multi_cache.get(key)
    if hit is not None:
        #print("multi_step_matrix cache hit:", key)
        return hit
    #print("multi_step_matrix cache miss:", key)
    M = np.eye(c_start+1)
    c = c_start
    if delta > 0:
        for _ in range(delta):
            K = gain_matrix(c)
            M = M @ K
            c += 1
    elif delta < 0:
        m = -delta
        for _ in range(m):
            K = loss_matrix(c)
            M = M @ K
            c -= 1
            if c < 0:
                raise ValueError("Loss drove CN below 0")
    _multi_cache[key] = (M, c)
    return M, c


def clear_multi_step_cache():
    """Clear the global cache for multi_step_matrix."""
    _multi_cache.clear()


# For a given child and segment get the edge matrix distribution based
# on the possible paths for copy number changes
def edge_matrix_from_diff(child, seg: int, c_parent: int):
    c_child = int(child.props['CN_profile'][seg]['cn_tot'])
    cache = child.props.setdefault('edge_M_cache', {})
    key = (seg, c_parent, c_child)                  # <-- use child CN
    hit = cache.get(key)
    if hit is not None:
        return hit

    delta = c_child - c_parent                      # derive, don’t trust CN_diff
    if delta == 0:
        M = np.eye(c_parent + 1)
        cache[key] = (M, c_child)
        return M, c_child

    M, c_end = multi_step_matrix(c_parent, delta)
    if c_end != c_child:
        raise ValueError(f"Edge CN mismatch seg {seg}: parent {c_parent} + {delta} -> {c_end}, child says {c_child}")
    cache[key] = (M, c_child)
    return M, c_child


# Clear per-edge caches
def clear_edge_caches(tree):
    for n in tree.traverse():
        n.props.pop('edge_M_cache', None)
        n.props.pop('edge_logM_cache', None)


# Safe logM so that log(0) gives -inf without error
def safe_logM(M):
    result = np.full_like(M, NEG_INF, dtype=float)
    mask = M > 0
    result[mask] = np.log(M[mask])
    return result


########################################
########################################
# EVERYTHING BELOW HERE HAS BEEN DEPRECATED
# AFTER REBUILD WITH VECTORIZATION
########################################
########################################


# Precompute functions for logM matrices per edge and segment
# To use (before locus_loglik_ab):
# segments = segments_from_snvs(snv_dict)
# precompute_logM_cache(tree, segments)
# get_logM = make_get_logM_from_precomputed()

# Get all CNV segments from SNV dict so we only use segments we need
def segments_from_snvs(snv_dict):
    segs = set()
    for rec in snv_dict.values():
        seg = rec.get("cna_idx", None)
        if seg is not None:
            segs.add(int(seg))
    return sorted(segs)

# Group SNVs by their CNA segment
def group_snvs_by_segment(snv_ids, snv_dict):
    seg2snvs = defaultdict(list)
    for snv_id in snv_ids:
        seg = snv_dict[snv_id]["cna_idx"]
        if seg is not None:
            seg2snvs[int(seg)].append(snv_id)
    return seg2snvs

# Precompute logM matrices for all edges and segments -
# turns the per-edge per-(c_parent,c_child) cache into a per-edge per-segment cache
def precompute_logM_cache(tree, segments):
    """
    For each child node and segment, precompute the log transition matrix.
    Stores: child.props['edge_logM_pre'][seg] = (logM, c_parent, c_child)
    """
    for n in tree.traverse():
        if 'edge_logM_pre' not in n.props:
            n.add_prop('edge_logM_pre', {})

    for seg in segments:
        for child in tree.traverse():
            if child.is_root:
                continue
            parent = child.up
            c_parent = int(parent.props['CN_profile'][seg]['cn_tot'])
            delta = int(child.props['CN_diff'][seg])
            M, c_child = multi_step_matrix(c_parent, delta)
            logM = safe_logM(M)
            child.props['edge_logM_pre'][seg] = (logM, c_parent, c_child)

# turn the logM calculation in the locus likelihood into a simple lookup
def make_get_logM_from_precomputed():
    def _get_logM(child, seg, c_parent_expect=None):
        logM, c_parent, c_child = child.props['edge_logM_pre'][seg]
        if c_parent_expect is not None and c_parent_expect != c_parent:
            # sanity guard – if this ever triggers, CN_profile changed after precompute
            raise AssertionError(f"Parent CN mismatch for seg={seg}: expected {c_parent_expect}, precomputed {c_parent}")
        return logM
    return _get_logM

# return flag if leaf is covered for SNV
def _leaf_has_cov(leaf, snv_id):
    k = int(leaf.props['alt'][snv_id])
    r = int(leaf.props['ref'][snv_id])
    return (k + r) > 0

# Build induced tree based on leaves with coverage
# avoids processing full tree when leaves have no coverage
def build_active_nodes_for_snv(tree, snv_id):
    covered = [leaf for leaf in tree.leaves() if _leaf_has_cov(leaf, snv_id)]
    if not covered:
        # No coverage anywhere → present channel is neutral; still need absent/null
        return {tree}

    active = set(covered)

    # Add LCAs of covered leaves
    stack = covered[:]
    while len(stack) > 1:
        a = stack.pop()
        b = stack.pop()
        lca = tree.common_ancestor(a, b)
        if lca not in active:
            active.add(lca)
        stack.append(lca)

    # Add all ancestors to root (closure)
    for v in list(active):
        cur = v
        while cur is not None:
            if cur in active:
                cur = cur.up
                continue
            active.add(cur)
            cur = cur.up
    return active


