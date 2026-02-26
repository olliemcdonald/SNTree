# transition_model.py

from dataclasses import dataclass
import numpy as np

@dataclass
class TransitionModel:
    """
    Stores precomputed logM transition matrices for each edge and segment.
    """

    # logM[e][s] → (parent_dim, child_dim) ndarray
    logM: list           # list of dicts: logM[edge_id][seg] = matrix

    parent_CN: np.ndarray   # (N_nodes, N_segments)
    child_CN: np.ndarray    # (N_nodes, N_segments)

    edge_parent: np.ndarray  # (N_edges,)
    edge_child: np.ndarray   # (N_edges,)

    n_edges: int

    def get(self, edge_id, seg):
        """
        Retrieve logM for an edge and segment.
        """
        return self.logM[edge_id][seg]