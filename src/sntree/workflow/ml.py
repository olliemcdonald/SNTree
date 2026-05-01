# sntree/workflow/ml.py

import os
import time
import pickle
import pandas as pd
from cyvcf2 import VCF

from sntree.io.io_tree import read_preprocessed_tree
from sntree.io.io_cna import import_cna_data, add_cna, cna_lookups, add_cna_bins
from sntree.io.io_snv import vcf_list_to_tables, snv_lookups
from sntree.io.io_preprocess import build_all
from sntree.likelihood.loglik_all_snvs import loglik_all_snvs


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def run_ml(sample, output_root, config, input_paths):

    sample_base = os.path.join(output_root, sample, "sntree")
    sample_out = os.path.join(sample_base, "ml")
    os.makedirs(sample_out, exist_ok=True)

    print(f"[{now()}] ML stage started for sample {sample}")
    t0 = time.time()

    # ---- Tree ----
    print(f"[{now()}] Loading preprocessed CNA tree...")
    if not os.path.exists(input_paths.preprocessed_tree):
        raise RuntimeError(
            f"Preprocessed tree not found at {input_paths.preprocessed_tree}. "
            "Run 'sntree preprocess' first."
        )

    t = read_preprocessed_tree(input_paths.preprocessed_tree)

    # ---- CNA ----
    print(f"[{now()}] Loading CNA profiles...")
    sample_mapping, cna_profiles = import_cna_data(
        input_paths.sample_mapping,
        input_paths.cna_profiles,
    )
    cna_idx, _ = cna_lookups(cna_profiles)
    cna_profiles = add_cna_bins(cna_profiles, cna_idx)
    t = add_cna(t, sample_mapping, cna_profiles)

    # ---- SNV ----
    print(f"[{now()}] Loading SNVs...")
    vcf_list = VCF(input_paths.vcf)
    variant_ids, ref_df, alt_df, normal_ref, normal_alt = vcf_list_to_tables(
        vcf_list, min_cells=2, normal_name=input_paths.normal_name
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
        pi0=config.pi0,
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
