# io_preprocess.py
import numpy as np # type: ignore
from sntree.model.tree_model import CNATree
from sntree.model.snv_dataset import SNVDataset
from sntree.model.transition_model import TransitionModel
from sntree.phylo.phylo import multi_step_matrix, safe_logM

def build_cna_tree(ete_tree, cna_profiles, sample_mapping):
    """
    Convert an ETE tree with CN profiles stored in node.props
    into a CNATree with pure NumPy arrays.
    """
    # ---- 1. Get a stable list of nodes (preorder) ----
    ete_nodes = list(ete_tree.traverse("preorder"))
    n_nodes = len(ete_nodes)

    # Map ETE nodes -> idx
    ete_to_idx = {node: i for i, node in enumerate(ete_nodes)}
    idx_to_ete = ete_nodes

    # ---- 2. Build parent and children arrays ----
    parent = np.full(n_nodes, -1, dtype=int)
    children = [[] for _ in range(n_nodes)]

    for node in ete_nodes:
        i = ete_to_idx[node]
        if node.up is not None:
            parent[i] = ete_to_idx[node.up]
        for ch in node.children:
            children[i].append(ete_to_idx[ch])

    # ---- 3. Identify leaves ----
    is_leaf = np.array([node.is_leaf for node in ete_nodes], dtype=bool)

    leaf_nodes = [node for node in ete_nodes if node.is_leaf]
    n_leaves = len(leaf_nodes)

    leaf_order = np.repeat(-1, n_leaves)
    node_of_leaf = np.full(n_nodes, -1, dtype=int)

    for leaf_idx, node in enumerate(leaf_nodes):
        node_of_leaf[ete_to_idx[node]] = leaf_idx


    leaf_to_node = np.full(n_leaves, -1, dtype=int)
    for node_idx in range(n_nodes):
        lf = node_of_leaf[node_idx]
        if lf != -1:
            leaf_to_node[lf] = node_idx

    # ---- 4. Gather CNA segments ----
    segments = sorted(list(set(cna_profiles["idx"].astype(int))))
    n_segments = len(segments)

    # ---- 5. CN matrix ----
    CN = np.zeros((n_nodes, n_segments), dtype=int)
    for node in ete_nodes:
        i = ete_to_idx[node]
        prof = node.props["CN_profile"]
        for seg_idx, d in prof.items():
            CN[i, int(seg_idx)] = int(d["cn_tot"])

    # ---- 6. traversal orders ----
    preorder = np.array([ete_to_idx[n] for n in ete_tree.traverse("preorder")], dtype=int)
    postorder = np.array([ete_to_idx[n] for n in ete_tree.traverse("postorder")], dtype=int)

    # ---- 7. depth ----
    depth = np.zeros(n_nodes, dtype=int)
    for node in ete_nodes:
        i = ete_to_idx[node]
        d = 0
        cur = node
        while cur.up is not None:
            d += 1
            cur = cur.up
        depth[i] = d

    # ---- 8. PRECOMPUTE leaf-descendant mask ----
    L = n_leaves
    leaf_desc_mask = np.zeros((n_nodes, L), dtype=bool)

    # bottom-up DP
    for u in postorder:
        # mark if this node is a leaf
        leaf_idx = node_of_leaf[u]
        if leaf_idx != -1:
            leaf_desc_mask[u, leaf_idx] = True
        # OR children’s masks
        for ch in children[u]:
            leaf_desc_mask[u] |= leaf_desc_mask[ch]

    # ---- 9. Construct CNATree ----
    return CNATree(
        parent=parent,
        children=children,
        is_leaf=is_leaf,
        leaf_order=leaf_order,
        node_of_leaf=node_of_leaf,
        leaf_to_node=leaf_to_node,
        CN=CN,
        postorder=postorder,
        preorder=preorder,
        depth=depth,
        ete_to_idx=ete_to_idx,
        idx_to_ete=idx_to_ete,
        leaf_desc_mask=leaf_desc_mask
    )


def build_snv_dataset(ref_df, alt_df, snv_df, tree_leaf_names):
    """
    Convert SNV read DataFrames and SNV metadata into SNVDataset.
    """
    # 1. Filter SNV columns to tree leaves
    ref_df_filtered = ref_df[tree_leaf_names]
    alt_df_filtered = alt_df[tree_leaf_names]
    leaf_names = np.array(tree_leaf_names)

    # ---- 2. SNV order is simply the index of ref_df / alt_df ----
    snv_ids = np.array(ref_df_filtered.index)

    # ---- 3. Convert matrices ----
    ks = alt_df_filtered.to_numpy(dtype=int)
    ns = (alt_df_filtered + ref_df_filtered).to_numpy(dtype=int)


    # ---- 4. Segments per SNV from snv_df ----
    seg_series = snv_df.set_index("snv")["cna_idx"]
    seg = seg_series.loc[snv_ids].astype(int).to_numpy()

    # ---- 5. Group SNVs by segment ----
    snvs_by_seg = {}
    for s in np.unique(seg):
        snvs_by_seg[int(s)] = np.where(seg == s)[0]

    return SNVDataset(
        ks=ks,
        ns=ns,
        seg=seg,
        snv_ids=snv_ids,
        leaf_names=leaf_names,
        snvs_by_seg=snvs_by_seg
    )


def build_transition_model(tree, segments):
    """
    Precompute logM for every edge and segment.
    tree: CNATree
    """
    n_nodes = tree.n_nodes
    n_segments = len(segments)

    # ---- 1. Enumerate edges ----
    edge_parent = []
    edge_child = []
    for parent_idx, ch_list in enumerate(tree.children):
        for ch in ch_list:
            edge_parent.append(parent_idx)
            edge_child.append(ch)

    edge_parent = np.array(edge_parent, dtype=int)
    edge_child = np.array(edge_child, dtype=int)
    n_edges = len(edge_parent)

    # ---- 2. Build logM list: logM[e][seg] ----
    logM = [dict() for _ in range(n_edges)]

    parent_CN = tree.CN.copy()
    child_CN = tree.CN.copy()

    # ---- 3. Fill transition matrices and edge lookup ----
    edge_id = [dict() for _ in range(n_nodes)]
    for e in range(n_edges):
        p = edge_parent[e]
        c = edge_child[e]
        edge_id[p][c] = e
        for seg in segments:
            c_parent = parent_CN[p, seg]
            c_child = child_CN[c, seg]
            delta = int(c_child - c_parent)

            # Use your multi-step CN → CN transition model
            M, _ = multi_step_matrix(c_parent, delta)
            logM[e][seg] = safe_logM(M)    
        

    transitions =  TransitionModel(
        logM=logM,
        parent_CN=parent_CN,
        child_CN=child_CN,
        edge_parent=edge_parent,
        edge_child=edge_child,
        n_edges=n_edges
    )

    return transitions, edge_id


def build_all(ete_tree, cna_profiles, sample_mapping,
              ref_df, alt_df, snv_df):
    tree = build_cna_tree(ete_tree, cna_profiles, sample_mapping)
    tree_leaf_names = [node.props["cell_id"] for node in tree.idx_to_ete if node.is_leaf]
    dataset = build_snv_dataset(ref_df, alt_df, snv_df, tree_leaf_names)

    segments = sorted(dataset.snvs_by_seg.keys())
    transitions, edge_id = build_transition_model(tree, segments)
    tree.edge_id = edge_id

    # IMPORTANT: map leaf order in tree to dataset leaf order
    # dataset.leaf_names contains ref_df column names
    leaf_name_to_index = {name: i for i, name in enumerate(dataset.leaf_names)}

    leaf_order = np.full(tree.n_leaves, -1, dtype=int)
    for node_idx, leaf_idx in enumerate(tree.node_of_leaf):
        if leaf_idx != -1:
            ete_leaf = tree.idx_to_ete[node_idx]
            cell_name = ete_leaf.props.get("cell_id")
            if cell_name not in leaf_name_to_index:
                raise ValueError(f"Leaf {cell_name} missing from SNV dataset")
            leaf_order[leaf_idx] = leaf_name_to_index[cell_name]

    tree.leaf_order = leaf_order

    return tree, dataset, transitions