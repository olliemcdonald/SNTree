# sntree/workflow/refine.py

import os
import time
import pandas as pd

from sntree.io.io_tree import read_preprocessed_tree
from sntree.io.io_cna import import_cna_data, add_cna, cna_lookups, add_cna_bins, barcode_cell_maps
from sntree.io.snv_index import build_snv_index
from sntree.phylo.refinement import refine_all_groups


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def run_refine(
    sample,
    input_root,
    output_root,
    placements,
    alpha,
    beta,
    config
):

    sample_base = os.path.join(output_root, sample, "sntree")
    sample_out = os.path.join(sample_base, "refine")
    os.makedirs(sample_out, exist_ok=True)

    print(f"[{now()}] Refinement stage started for sample {sample}")
    t0 = time.time()

    # ---- Paths ----
    tree_path = os.path.join(sample_base, "tree_preprocessed.new")
    sample_map_file = f"{input_root}/{sample}/chisel/{sample}.info.tsv"
    cna_file = f"{input_root}/{sample}/medicc2/{sample}_final_cn_profiles.tsv"
    cna_events_file = f"{input_root}/{sample}/medicc2/{sample}_copynumber_events_df.tsv"
    cna_distance_file = f"{input_root}/{sample}/medicc2/{sample}_pairwise_distances.tsv"
    vcf_path = f"{input_root}/{sample}/snv/consensus_singlecell_counts.vcf.gz"

    # ---- Load preprocessed tree ----
    print(f"[{now()}] Loading preprocessed CNA tree...")
    if not os.path.exists(tree_path):
        raise RuntimeError(
            f"Preprocessed tree not found at {tree_path}. "
            "Run 'sntree preprocess' first."
        )

    tree = read_preprocessed_tree(tree_path)

    # ---- Load CNA ----
    print(f"[{now()}] Loading CNA profiles...")
    sample_mapping, cna_profiles, _ = import_cna_data(
        sample_map_file,
        cna_file,
        cna_events_file
    )

    cna_idx, _ = cna_lookups(cna_profiles)
    cna_profiles = add_cna_bins(cna_profiles, cna_idx)
    tree = add_cna(tree, sample_mapping, cna_profiles)

    # ---- Name mapping ----
    barcode_to_cell, _ = barcode_cell_maps(sample_mapping)

    # ---- SNV index ----
    print(f"[{now()}] Building SNV index...")
    snv_index = build_snv_index(
        vcf_path,
        normal_name=None,
        name_map=barcode_to_cell,
    )

    # ---- Refinement ----
    print(f"[{now()}] Running subtree refinement...")
    results, final_assignments = refine_all_groups(
        tree=tree,
        snv_assignments=placements,
        cna_profiles=cna_profiles,
        sample_mapping=sample_mapping,
        cna_distance_tsv=cna_distance_file,
        vcf_path=vcf_path,
        snv_index=snv_index,
        output_dir=sample_out,
        alpha=alpha,
        beta=beta,
        p0=config.p0,
        batch_size=config.batch_size,
        max_iters=config.nni_max_iters,
        init="nj",
    )

    runtime = time.time() - t0

    print(f"[{now()}] Refinement complete.")
    print(f"[{now()}] Runtime: {runtime:.2f} sec")
    print(f"[{now()}] Refined groups: {len(results)}")

    # ---- Write final full tree ----
    refined_tree_path = os.path.join(sample_out, "refined_full_tree.new")
    with open(refined_tree_path, "w") as f:
        f.write(tree.write(parser=1).strip() + "\n")

    # ---- Write summary ----
    summary_df = pd.DataFrame([{
        "sample": sample,
        "alpha_used": alpha,
        "beta_used": beta,
        "p0": config.p0,
        "batch_size": config.batch_size,
        "max_iters": config.nni_max_iters,
        "n_groups_refined": len(results),
        "runtime_sec": runtime
    }])

    summary_df.to_csv(
        os.path.join(sample_out, "refine_summary.tsv"),
        sep="\t",
        index=False
    )

    # ---- Save final SNV assignments ----
    final_df = pd.DataFrame.from_dict(
        final_assignments, orient="index", columns=["node"]
    )
    final_df.index.name = "snv"
    final_df.to_csv(
        os.path.join(sample_out, "snv_assignments_final.tsv"),
        sep="\t"
    )

    print(f"[{now()}] Refinement stage finished successfully.")

    return results