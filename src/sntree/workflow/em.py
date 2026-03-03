# sntree/workflow/em.py

import os
import time
import pickle
import pandas as pd
from cyvcf2 import VCF

from sntree.io.io_tree import read_preprocessed_tree
from sntree.io.io_cna import import_cna_data, add_cna, cna_lookups, add_cna_bins
from sntree.io.io_snv import vcf_list_to_tables, snv_lookups
from sntree.io.io_preprocess import build_all
from sntree.likelihood.em_alpha_beta import em_alpha_beta


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def run_em(sample, input_root, output_root, config):

    sample_base = os.path.join(output_root, sample, "sntree")
    sample_out = os.path.join(sample_base, "em")
    os.makedirs(sample_out, exist_ok=True)

    print(f"[{now()}] EM stage started for sample {sample}")
    t0 = time.time()

    # ---- Paths ----
    tree_path = os.path.join(sample_base, "tree_preprocessed.new")
    sample_map_file = f"{input_root}/{sample}/chisel/{sample}.info.tsv"
    cna_file = f"{input_root}/{sample}/medicc2/{sample}_final_cn_profiles.tsv"
    cna_events_file = f"{input_root}/{sample}/medicc2/{sample}_copynumber_events_df.tsv"
    vcf_path = f"{input_root}/{sample}/snv/consensus_singlecell_counts.vcf.gz"
    normal_name = f"{input_root}/{sample}/normal_cells/{sample}_normal_markdup.bam"

    # ---- Tree ----
    print(f"[{now()}] Loading preprocessed CNA tree...")
    if not os.path.exists(tree_path):
        raise RuntimeError(
            f"Preprocessed tree not found at {tree_path}. "
            "Run 'sntree preprocess' first."
        )

    t = read_preprocessed_tree(tree_path)

    # ---- CNA ----
    print(f"[{now()}] Loading CNA profiles...")
    sample_mapping, cna_profiles, cna_events = import_cna_data(
        sample_map_file, cna_file, cna_events_file
    )
    cna_idx, _ = cna_lookups(cna_profiles)
    cna_profiles = add_cna_bins(cna_profiles, cna_idx)
    t = add_cna(t, sample_mapping, cna_profiles)

    # ---- SNV ----
    print(f"[{now()}] Loading SNVs...")
    vcf_list = VCF(vcf_path)
    variant_ids, ref_df, alt_df, normal_ref, normal_alt = vcf_list_to_tables(
        vcf_list, min_cells=2, normal_name=normal_name
    )

    snv_df, snv_dict, ref_df, alt_df, normal_ref, normal_alt = snv_lookups(
        variant_ids,
        cna_idx,
        ref_df=ref_df,
        alt_df=alt_df,
        normal_ref=normal_ref,
        normal_alt=normal_alt
    )

    # ---- Build unified structures ----
    print(f"[{now()}] Building data structures...")
    cna_tree, snv_dataset, transitions = build_all(
        ete_tree=t,
        cna_profiles=cna_profiles,
        sample_mapping=sample_mapping,
        ref_df=ref_df,
        alt_df=alt_df,
        snv_df=snv_df
    )

    # ---- EM Algorithm ----
    print(f"[{now()}] Running EM algorithm...")
    alpha, beta, placements_em, history = em_alpha_beta(
        cna_tree,
        snv_dataset,
        transitions,
        init_alpha=config.alpha_init,
        init_beta=config.beta_init,
        p0=config.p0,
        pi0=0.0,
        p1_fp_mode="one_over_c",
        max_iter=config.em_max_iter,
        tol=1e-4,
        damp=0.5,
        min_alt_reads=2,
        min_alt_cells=2,
        batch_size=config.batch_size,
        print_progress=True
    )

    print(f"[{now()}] EM complete.")
    print(f"[{now()}] Estimated alpha: {alpha:.6f}")
    print(f"[{now()}] Estimated beta:  {beta:.6f}")
    print(f"[{now()}] Runtime: {time.time() - t0:.2f} sec")

    # ---- Convert placements to named dictionary ----
    placements_named = {}
    likelihoods = {}

    for var, (node_idx, ll) in placements_em.items():
        if node_idx is not None:
            node_name = cna_tree.idx_to_ete[node_idx].name
        else:
            node_name = "Null"

        placements_named[var] = node_name
        likelihoods[var] = float(ll)

    # ---- Save binary results ----
    with open(os.path.join(sample_out, "em_results.pkl"), "wb") as f:
        pickle.dump({
            "alpha": alpha,
            "beta": beta,
            "placements": placements_named,
            "loglikelihoods": likelihoods,
            "history": history
        }, f)

    # ---- Save readable placements ----
    print(f"[{now()}] Writing EM placements to TSV...")
    placements_df = pd.DataFrame.from_dict(
        placements_named, orient="index", columns=["node"]
    )
    placements_df.index.name = "snv"
    placements_df.to_csv(
        os.path.join(sample_out, "placements_em.tsv"),
        sep="\t"
    )

    # ---- Save EM per-SNV likelihoods ----
    ll_df = pd.DataFrame.from_dict(
        likelihoods, orient="index", columns=["loglik"]
    )
    ll_df.index.name = "snv"
    ll_df.to_csv(
        os.path.join(sample_out, "placements_em_loglik.tsv"),
        sep="\t"
    )

    print(f"[{now()}] EM stage finished successfully.")

    return {
        "placements": placements_named,
        "alpha": alpha,
        "beta": beta
    }