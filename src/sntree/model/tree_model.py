# tree_model.py

from dataclasses import dataclass, field
import numpy as np

@dataclass
class CNATree:
    """
    Array-backed tree representation for fast likelihood/DP.
    """

    # Topology
    parent: np.ndarray              # shape (N_nodes,), parent index or -1
    children: list                  # list of lists: children[i] = list of child indices
    is_leaf: np.ndarray             # bool array shape (N_nodes,)
    leaf_order: np.ndarray          # index → node index
    node_of_leaf: np.ndarray        # node index → leaf index or -1

    # CN data
    CN: np.ndarray                  # shape (N_nodes, N_segments)

    # Traversal helpers
    postorder: np.ndarray
    preorder: np.ndarray
    depth: np.ndarray

    # Mapping ETE <-> indices
    ete_to_idx: dict               # ETE node → int
    idx_to_ete: list               # int → ETE node

    edge_id: list = field(default_factory=list)  # list of dicts: edge_id[i][child] = eid

    leaf_desc_mask: np.ndarray = field(default=None)  # (N_nodes, N_leaves)
    leaf_to_node: np.ndarray = field(default=None) # shape (n_leaves,)

    @property
    def n_nodes(self):
        return self.parent.shape[0]

    @property
    def n_leaves(self):
        return int(self.is_leaf.sum())

    @property
    def n_segments(self):
        return self.CN.shape[1]
    
