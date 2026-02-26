# snv_dataset.py

from dataclasses import dataclass
import numpy as np

@dataclass
class SNVDataset:
    """
    Dense SNV × leaf read matrices and per-SNV metadata.
    """

    ks: np.ndarray          # shape (N_snvs, N_leaves)
    ns: np.ndarray          # shape (N_snvs, N_leaves)
    seg: np.ndarray         # shape (N_snvs,) integers

    snv_ids: np.ndarray     # length N_snvs (list or array of strings)
    leaf_names: np.ndarray  # length N_leaves

    snvs_by_seg: dict       # seg → np.ndarray of SNV indices

    @property
    def n_snvs(self):
        return self.ks.shape[0]

    @property
    def n_leaves(self):
        return self.ks.shape[1]