# sntree/workflow/soft_em.py

import os
import time
import pickle
import numpy as np
import pandas as pd
from cyvcf2 import VCF

from sntree.io.io_tree import read_preprocessed_tree
from sntree.io.io_cna import import_cna_data, add_cna, cna_lookups, add_cna_bins
from sntree.io.io_snv import vcf_list_to_tables, snv_lookups
from sntree.io.io_preprocess import build_all
from sntree.likelihood.em_soft import em_soft
from sntree.likelihood.locus_loglik_batch import locus_loglik_batch
from sntree.constants import MU_ERR, EPS


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def compute_map_placements(
    cna_tree,
    snv_dataset,
    transitions,
    alpha,
    beta,
    pi_b,
    pi_0,
    p0=MU_ERR,
    p1_fp_mode="one_over_c",
    batch_size=1024,
):
    """
    Derive MAP (hard) placements from converged soft EM parameters.

    For each locus the MAP assignment is:
        argmax over {null, branches} of  log π_b + log L_m(b | α, β)

    Returns
    -------
    dict  {snv_id: node_name}   (null placements map to "Null")
    """
    log_pi_b  = np.log(np.maximum(pi_b, EPS))
    log_pi_0  = np.log(max(pi_0, EPS))
    log_1mpi0 = np.log(max(1.0 - pi_0, EPS))

    placements = {}

    for seg, snv_idx_list in snv_dataset.snvs_by_seg.items():
        for i0 in range(0, len(snv_idx_list), batch_size):
            batch = snv_idx_list[i0 : i0 + batch_size]

            logL_nodes, logL_null = locus_loglik_batch(
                cna_tree, snv_dataset, transitions, batch,
                alpha=alpha, beta=beta, p0=p0,
                p1_fp_mode=p1_fp_mode,
            )

            # Log-scores: (N+1, B_batch) with null in row 0
            log_null_score   = log_pi_0  + logL_null               # (B_batch,)
            log_branch_score = log_1mpi0 + log_pi_b[:, None] + logL_nodes  # (N, B_batch)

            for j, snv_idx in enumerate(batch):
                snv_id = snv_dataset.snv_ids[snv_idx]
                branch_scores = log_branch_score[:, j]
                null_score    = log_null_score[j]

                best_branch_idx = int(np.argmax(branch_scores))
                if branch_scores[best_branch_idx] > null_score:
                    node_name = cna_tree.idx_to_ete[best_branch_idx].name
                else:
                    node_name = "Null"

                placements[snv_id] = node_name

    return placements


def run_soft_em(
    sample,
    output_root,
    config,
    input_paths,
    init_alpha=None,
    init_beta=None,
    init_pi_b=None,
    tree_path_override=None,
    output_subdir="soft_em",
    joint=None,
    pass_label="pass 1",
    warm_start_pkl=None,
):
    """
    Standalone soft EM workflow stage.

    Parameters
    ----------
    init_alpha, init_beta : float or None
        Starting error parameters.  Defaults to config.alpha_init/beta_init.
    init_pi_b : np.ndarray or None
        Warm-start branch proportions (e.g. from a previous pass).
        If None and warm_start_pkl is also None, initialises uniformly.
    tree_path_override : str or None
        Path to a Newick tree to use instead of input_paths.preprocessed_tree.
        Used for pass 2 to load the NNI-refined tree.
    output_subdir : str
        Subdirectory name under {output_root}/{sample}/sntree/ for outputs.
    joint : bool or None
        Whether to jointly update alpha/beta during EM.
        If None: True when no hard EM results exist (pass 1, no hard EM),
        False otherwise.
    pass_label : str
        Display label for log messages.
    warm_start_pkl : str or None
        Path to a previous soft_em_results.pkl.  pi_b and alpha/beta are
        loaded from it to warm-start this run (used by the CLI soft-em command).

    Returns
    -------
    dict with keys: alpha, beta, pi_b, pi_0, placements, node_names
    """
    sample_base = os.path.join(output_root, sample, "sntree")
    sample_out  = os.path.join(sample_base, output_subdir)
    os.makedirs(sample_out, exist_ok=True)

    print(f"[{now()}] ========================================")
    print(f"[{now()}] Soft EM ({pass_label}) for {sample}")
    print(f"[{now()}] ========================================")
    t0 = time.time()

    # ── Warm start from pickle ─────────────────────────────────────────────
    if warm_start_pkl is not None:
        print(f"[{now()}] Loading warm-start from {warm_start_pkl}")
        with open(warm_start_pkl, "rb") as f:
            ws = pickle.load(f)
        if init_alpha is None:
            init_alpha = float(ws["alpha"])
        if init_beta is None:
            init_beta  = float(ws["beta"])
        if init_pi_b is None:
            init_pi_b  = ws["pi_b"]          # array; matched by name below
        warm_node_names = ws.get("node_names", None)
    else:
        warm_node_names = None

    init_alpha = float(init_alpha) if init_alpha is not None else config.alpha_init
    init_beta  = float(init_beta)  if init_beta  is not None else config.beta_init

    # ── Load tree ──────────────────────────────────────────────────────────
    tree_path = tree_path_override or input_paths.preprocessed_tree
    print(f"[{now()}] Loading tree from {tree_path}")
    if not os.path.exists(tree_path):
        raise RuntimeError(
            f"Tree not found at {tree_path}. "
            "Run 'sntree preprocess' (and 'sntree refine' for pass 2) first."
        )
    t = read_preprocessed_tree(tree_path)

    # ── Load CNA ───────────────────────────────────────────────────────────
    print(f"[{now()}] Loading CNA profiles...")
    sample_mapping, cna_profiles = import_cna_data(
        input_paths.sample_mapping,
        input_paths.cna_profiles,
    )
    cna_idx, _ = cna_lookups(cna_profiles)
    cna_profiles = add_cna_bins(cna_profiles, cna_idx)
    t = add_cna(t, sample_mapping, cna_profiles)

    # ── Load SNVs ──────────────────────────────────────────────────────────
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

    # ── Build unified structures ───────────────────────────────────────────
    print(f"[{now()}] Building data structures...")
    cna_tree, snv_dataset, transitions = build_all(
        ete_tree=t,
        cna_profiles=cna_profiles,
        sample_mapping=sample_mapping,
        ref_df=ref_df,
        alt_df=alt_df,
        snv_df=snv_df,
    )

    node_names = [cna_tree.idx_to_ete[i].name for i in range(cna_tree.n_nodes)]

    # ── Match warm-start pi_b to the current tree by node name ────────────
    if init_pi_b is not None and warm_node_names is not None:
        name_to_pi = dict(zip(warm_node_names, init_pi_b))
        matched = np.array([name_to_pi.get(n, np.nan) for n in node_names])
        missing_mask = np.isnan(matched)
        if missing_mask.any():
            # Unmatched nodes (new within-clade branches from NNI): use mean
            # of matched values so normalisation stays sensible
            matched[missing_mask] = np.nanmean(matched) if not np.all(missing_mask) else 1.0
        s = matched.sum()
        init_pi_b = matched / s if s > 0 else None
        n_matched = int((~missing_mask).sum())
        print(f"[{now()}] Warm-start: matched {n_matched}/{len(node_names)} nodes by name")

    # ── Infer joint flag ───────────────────────────────────────────────────
    if joint is None:
        # If we're estimating alpha/beta from scratch (no hard EM), use joint
        hard_em_pkl = os.path.join(sample_base, "em", "em_results.pkl")
        joint = not os.path.exists(hard_em_pkl)
    if joint:
        print(f"[{now()}] Joint mode: alpha/beta estimated alongside pi_b")
    else:
        print(f"[{now()}] Fixed mode: alpha={init_alpha:.5f}  beta={init_beta:.5f}")

    # ── Run soft EM ────────────────────────────────────────────────────────
    print(f"[{now()}] Running soft EM ({pass_label})...")
    alpha, beta, pi_b, pi_0, history = em_soft(
        cna_tree,
        snv_dataset,
        transitions,
        init_alpha=init_alpha,
        init_beta=init_beta,
        init_pi0=config.pi0,
        init_pi_b=init_pi_b,
        alpha_dir=config.alpha_dir,
        p0=config.p0,
        p1_fp_mode="one_over_c",
        max_iter=config.soft_em_max_iter,
        tol=1e-4,
        joint=joint,
        batch_size=config.batch_size,
        print_progress=True,
    )

    print(f"[{now()}] Soft EM complete (runtime={time.time() - t0:.2f} sec)")
    print(f"[{now()}] alpha={alpha:.5f}  beta={beta:.5f}  pi_0={pi_0:.4f}")

    # ── Derive MAP placements (for refinement / downstream) ───────────────
    print(f"[{now()}] Computing MAP placements from soft assignments...")
    placements = compute_map_placements(
        cna_tree, snv_dataset, transitions,
        alpha=alpha, beta=beta,
        pi_b=pi_b, pi_0=pi_0,
        p0=config.p0,
        p1_fp_mode="one_over_c",
        batch_size=config.batch_size,
    )

    # ── Save branch proportions TSV ───────────────────────────────────────
    pi_df = pd.DataFrame({
        "node":    node_names,
        "pi_b":    pi_b,
        "is_leaf": cna_tree.is_leaf,
    }).sort_values("pi_b", ascending=False)
    pi_df.index.name = "node_idx"
    pi_df.to_csv(
        os.path.join(sample_out, "branch_proportions.tsv"),
        sep="\t", float_format="%.6f",
    )

    # ── Save MAP placements TSV ───────────────────────────────────────────
    placements_df = pd.DataFrame.from_dict(
        placements, orient="index", columns=["node"]
    )
    placements_df.index.name = "snv"
    placements_df.to_csv(
        os.path.join(sample_out, "placements_soft.tsv"), sep="\t"
    )

    # ── Save full pickle ───────────────────────────────────────────────────
    with open(os.path.join(sample_out, "soft_em_results.pkl"), "wb") as f:
        pickle.dump({
            "pi_b":       pi_b,
            "pi_0":       pi_0,
            "node_names": node_names,
            "alpha":      alpha,
            "beta":       beta,
            "history":    history,
        }, f)

    print(f"[{now()}] Results written to {sample_out}/")

    return {
        "alpha":      alpha,
        "beta":       beta,
        "pi_b":       pi_b,
        "pi_0":       pi_0,
        "placements": placements,
        "node_names": node_names,
    }
