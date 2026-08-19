# sntree/workflow/load_inputs.py

import os
import time

from cyvcf2 import VCF

from sntree.io.io_tree import read_preprocessed_tree
from sntree.io.io_cna import import_cna_data, add_cna, cna_lookups, add_cna_bins
from sntree.io.io_snv import vcf_list_to_tables, snv_lookups
from sntree.io.io_preprocess import build_all


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def load_structures(input_paths, tree_path, verbose=True):
    """
    Load the tree, CNA profiles and SNV read counts and build the unified
    (CNATree, SNVDataset, TransitionModel) structures.

    Parameters
    ----------
    input_paths : InputPaths
    tree_path : str
        Newick tree to use (preprocessed MEDICC2 tree, or a refined tree).

    Returns
    -------
    cna_tree, snv_dataset, transitions
    """
    if verbose:
        print(f"[{now()}] Loading tree from {tree_path}")
    if not os.path.exists(tree_path):
        raise RuntimeError(
            f"Tree not found at {tree_path}. "
            "Run 'sntree preprocess' (and 'sntree refine' for pass 2) first."
        )
    t = read_preprocessed_tree(tree_path)

    if verbose:
        print(f"[{now()}] Loading CNA profiles...")
    sample_mapping, cna_profiles = import_cna_data(
        input_paths.sample_mapping,
        input_paths.cna_profiles,
    )
    cna_idx, _ = cna_lookups(cna_profiles)
    cna_profiles = add_cna_bins(cna_profiles, cna_idx)
    t = add_cna(t, sample_mapping, cna_profiles)

    if verbose:
        print(f"[{now()}] Loading SNVs...")
    vcf_list = VCF(input_paths.vcf)
    variant_ids, ref_df, alt_df, normal_ref, normal_alt = vcf_list_to_tables(
        vcf_list, min_cells=2, normal_name=input_paths.normal_name
    )
    snv_df, snv_dict, ref_df, alt_df, normal_ref, normal_alt = snv_lookups(
        variant_ids, cna_idx,
        ref_df=ref_df, alt_df=alt_df,
        normal_ref=normal_ref, normal_alt=normal_alt,
    )

    if verbose:
        print(f"[{now()}] Building data structures...")
    cna_tree, snv_dataset, transitions = build_all(
        ete_tree=t,
        cna_profiles=cna_profiles,
        sample_mapping=sample_mapping,
        ref_df=ref_df,
        alt_df=alt_df,
        snv_df=snv_df,
    )

    return cna_tree, snv_dataset, transitions
