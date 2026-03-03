import pickle
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from cyvcf2 import VCF


@dataclass
class SNVIndex:
    """
    Lightweight SNV index:
      - snv_ids: list of "chrom:pos:ref>alt"
      - chroms, position, refs, alts: per SNV
      - alt_cells: list of np.ndarray of sample indices with alt>0
      - sample_names: sample names (after optional mapping)
    """
    snv_ids: List[str]
    chroms: List[str]
    position: List[int]
    refs: List[str]
    alts: List[str]
    alt_cells: List[np.ndarray]
    sample_names: List[str]

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "SNVIndex":
        with open(path, "rb") as f:
            return pickle.load(f)


def _map_sample_names(samples: Sequence[str], name_map: Optional[Dict[str, str]]) -> List[str]:
    if not name_map:
        return list(samples)
    return [name_map.get(s, s) for s in samples]


def build_snv_index(
    vcf_path: str,
    normal_name: Optional[str] = None,
    name_map: Optional[Dict[str, str]] = None,
    min_cells: int = 1,
) -> SNVIndex:
    """
    Build a minimal SNV index using cyvcf2 without loading full read matrices.
    """
    vcf = VCF(vcf_path)
    raw_samples = list(vcf.samples)

    # identify normal in raw samples
    if normal_name is not None and normal_name in raw_samples:
        normal_idx = raw_samples.index(normal_name)
    else:
        normal_idx = None

    # keep cell sample indices (exclude normal)
    cell_idx = [i for i in range(len(raw_samples)) if i != normal_idx]

    # map sample names if requested
    samples_mapped = _map_sample_names(raw_samples, name_map)
    cell_names = [samples_mapped[i] for i in cell_idx]

    snv_ids: List[str] = []
    chroms: List[str] = []
    position: List[int] = []
    refs: List[str] = []
    alts: List[str] = []
    alt_cells: List[np.ndarray] = []

    for rec in vcf:
        if not rec.REF or len(rec.REF) != 1:
            continue

        AD = rec.format("AD")
        if AD is None:
            continue
        if AD.ndim != 2 or AD.shape[0] != len(raw_samples) or AD.shape[1] < 2:
            continue

        n_alts = AD.shape[1] - 1
        for j, a in enumerate(rec.ALT or []):
            if (a is None) or (isinstance(a, str) and a.startswith("<") and a.endswith(">")) or len(a) != 1:
                continue
            if j >= n_alts:
                continue

            alt_col = AD[:, 1 + j]
            alt_in_cells = alt_col[cell_idx]
            support_mask = alt_in_cells > 0
            if int(np.sum(support_mask)) < min_cells:
                continue

            snv_ids.append(f"{rec.CHROM}:{rec.POS}:{rec.REF}>{a}")
            chroms.append(str(rec.CHROM))
            position.append(int(rec.POS))
            refs.append(str(rec.REF))
            alts.append(str(a))

            alt_cells_idx = np.where(support_mask)[0].astype(int)
            alt_cells.append(alt_cells_idx)

    return SNVIndex(
        snv_ids=snv_ids,
        chroms=chroms,
        position=position,
        refs=refs,
        alts=alts,
        alt_cells=alt_cells,
        sample_names=cell_names,
    )

def load_snv_assignments(tsv_path: str, skip: int = 0) -> Dict[str, str]:
    """
    Load SNV -> node assignment mapping.
    tsv_path : str
        Path to TSV file.
    skip : int
        Number of initial lines to skip (e.g. header rows).
    """
    mapping = {}
    with open(tsv_path, "r") as f:
        for i, line in enumerate(f):
            if i < skip:
                continue
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            snv_id, node = line.split("\t")[:2]
            mapping[snv_id] = node
    return mapping


def filter_snvs_for_group_from_assignments(
    index: SNVIndex,
    snv_to_node: Dict[str, str],
    group_node: Any,
    group_cells: Iterable[str],
    exclude_root: bool = True,
    exclude_null: bool = True,
) -> List[str]:
    """
    Select SNVs for a CNA-identical group using prior SNV placement results.

    Keeps SNVs assigned to:
      - the group MRCA node
      - any leaf (cell) in the group

    Assumes:
      - node labels for leaves match cell identifiers
      - group_node is the MRCA node (ete4 Tree node)
    """
    subtree_names = {n.name for n in group_node.traverse()}

    selected = []
    for snv_id in index.snv_ids:
        node_name = snv_to_node.get(snv_id)
        if node_name is None:
            continue

        node_name_lower = node_name.lower()
        if exclude_null and node_name_lower == "Null":
            continue
        if exclude_root and node_name_lower == "root":
            continue

        if node_name in subtree_names:
            selected.append(snv_id)

    return selected





def filter_snvs_for_group_from_alt_cells(
    index: SNVIndex,
    group_cells: Iterable[str],
    exclude_outside: bool = False,
) -> List[str]:
    """
    Select SNVs with any alt support in group_cells.
    If exclude_outside=True, require alt support only within the group.
    """
    group_set = set(group_cells)

    # map group cells to indices in index.sample_names
    name_to_idx = {n: i for i, n in enumerate(index.sample_names)}
    group_idx = {name_to_idx[n] for n in group_set if n in name_to_idx}

    if len(group_idx) == 0:
        return []

    selected = []
    group_idx_set = set(group_idx)
    for snv_id, alt_idx in zip(index.snv_ids, index.alt_cells):
        alt_idx_set = set(int(x) for x in alt_idx.tolist())
        if alt_idx_set & group_idx_set:
            if exclude_outside and not alt_idx_set.issubset(group_idx_set):
                continue
            selected.append(snv_id)
    return selected

