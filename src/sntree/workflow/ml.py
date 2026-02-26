# sntree/workflow/ml.py

import os
import time
import pickle
import pandas as pd
from cyvcf2 import VCF

from sntree.io.io_tree import read_tree
from sntree.io.io_cna import import_cna_data, add_cna, cna_lookups, add_cna_bins
from sntree.io.io_snv import vcf_list_to_tables, snv_lookups
from sntree.io.io_preprocess import build_all
from sntree.likelihood.loglik_all_snvs import loglik_all_snvs


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def run_ml(sample, input_root, output_root, config):

    sample_out = os.path.join(output_root, sample, "ml")
    os.makedirs(sample_out, exist_ok=True)

    print(f"[{now()}] ML stage started for sample {sample}")
    t0 = time.time()

    # ---- Paths ----
    medicc_newick_fn = f"{input_root}/{sample}/medicc2/{sample}_final_tree.new"
    sample_map_file = f"{input_root}/{sample}/chisel/{sample}.info.tsv"
    cna_file = f"{input_root}/{sample}/medicc2/{sample}_final_cn_profiles.tsv"
    cna_events_file = f"{input_root}/{sample}/medicc2/{sample}_copynumber_events_df.tsv"
    vcf_path = f"{input_root}/{sample}/snv/consensus_singlecell_counts.vcf.gz"
    normal_name = f"{input_root}/{sample}/normal_cells/{sample}_normal_markdup.bam"

    # ---- Tree ----
    print(f"[{now()}] Loading CNA tree...")
    t = read_tree(medicc_newick_fn, remove_diploid=True, diploid_name="diploid")

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

    # ---- Maximum Likelihood Placement ----
    print(f"[{now()}] Running CNA-aware SNV placement...")
    placements, total_ll = loglik_all_snvs(
        cna_tree,
        snv_dataset,
        transitions,
        alpha=config.alpha_init,
        beta=config.beta_init,
        p0=config.p0,
        pi0=0.0,
        p1_fp_mode="one_over_c",
        min_alt_reads=2,
        min_alt_cells=2,
        batch_size=config.batch_size
    )

    print(f"[{now()}] ML placement complete.")
    print(f"[{now()}] Total log-likelihood: {total_ll:.4f}")
    print(f"[{now()}] Runtime: {time.time() - t0:.2f} sec")

    # ---- Convert placements ----
    placements_named = {}
    likelihoods = {}

    for var, (node_idx, ll) in placements.items():
        if node_idx is not None:
            node_name = cna_tree.idx_to_ete[node_idx].name
        else:
            node_name = "Null"

        placements_named[var] = node_name
        likelihoods[var] = float(ll)

    # ---- Save binary results ----
    with open(os.path.join(sample_out, "ml_results.pkl"), "wb") as f:
        pickle.dump({
            "alpha": config.alpha_init,
            "beta": config.beta_init,
            "loglikelihood_total": total_ll,
            "placements": placements_named,
            "loglikelihoods": likelihoods
        }, f)

    # ---- Save readable placements ----
    print(f"[{now()}] Writing placements to TSV...")
    placements_df = pd.DataFrame.from_dict(
        placements_named, orient="index", columns=["node"]
    )
    placements_df.index.name = "snv"
    placements_df.to_csv(
        os.path.join(sample_out, "placements_ml.tsv"),
        sep="\t"
    )

    # ---- Save per-SNV likelihoods ----
    ll_df = pd.DataFrame.from_dict(
        likelihoods, orient="index", columns=["loglik"]
    )
    ll_df.index.name = "snv"
    ll_df.to_csv(
        os.path.join(sample_out, "placements_ml_loglik.tsv"),
        sep="\t"
    )

    # ---- Save summary ----
    summary_df = pd.DataFrame([{
        "sample": sample,
        "alpha": config.alpha_init,
        "beta": config.beta_init,
        "p0": config.p0,
        "batch_size": config.batch_size,
        "loglikelihood_total": total_ll
    }])

    summary_df.to_csv(
        os.path.join(sample_out, "ml_summary.tsv"),
        sep="\t",
        index=False
    )

    print(f"[{now()}] ML stage finished successfully.")

    return {
        "placements": placements_named,
        "alpha": config.alpha_init,
        "beta": config.beta_init,
        "loglikelihood": total_ll
    }