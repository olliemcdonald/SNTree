# sntree/workflow/soft_em.py

import os
import time
import pickle
import numpy as np
import pandas as pd

from sntree.workflow.load_inputs import load_structures
from sntree.likelihood.em_soft import em_soft
from sntree.likelihood.locus_loglik_batch import locus_loglik_batch
from sntree.constants import MU_ERR, EPS


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


#: Statuses recorded per locus in the ``score_status`` column.  Only "ok" loci
#: carry both a finite llr_null and a finite llr_margin and are therefore
#: eligible for LLR-based filtering; every other status must be handled
#: explicitly (and identically) in the observed and permuted passes, or the two
#: distributions stop being comparable.
SCORE_STATUS_OK              = "ok"               # both LLRs finite
SCORE_STATUS_NO_MARGIN       = "no_margin"        # < 2 finite branches, llr_null finite
SCORE_STATUS_NO_VALID_BRANCH = "no_valid_branch"  # every branch score -inf
SCORE_STATUS_NULL_IMPOSSIBLE = "null_impossible"  # null score -inf, best branch finite
SCORE_STATUS_UNDEFINED       = "undefined"        # -inf - -inf, i.e. NaN

PLACEMENT_COLUMNS = ["node", "llr_null", "llr_margin", "score_status"]


def score_placements(
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
    snv_mask=None,
    cell_perm_rng=None,
):
    """
    Score every locus under the converged soft EM parameters.

    One E-step pass.  For each locus this evaluates

        log_null_score   = log π_0     + log L_m(∅ | α)
        log_branch_score = log(1-π_0)  + log π_b + log L_m(b | α, β)

    and reports the MAP placement together with two confidence quantities:

        llr_null   = best_branch_score - null_score
                     how much better the best branch is than "this is noise"
        llr_margin = best_branch_score - second_best_branch_score
                     how confident the branch choice is

    The MAP placement itself is identical to what compute_map_placements has
    always produced, including the "Null" sentinel and its tie-breaking.

    Parameters
    ----------
    snv_mask : bool array (n_snvs,) or None
        Restrict scoring to a subset of loci (e.g. a subsample for a
        permutation null).  None scores every locus.
    cell_perm_rng : np.random.Generator or None
        When given, a fresh cell → tip permutation is drawn per locus batch and
        applied to the read counts (see locus_loglik_batch's cell_perm).  A real
        mutation's alt cells sit in one clade and lose their concordance under
        the shuffle; a scattered artefact is unaffected.  Redrawing per batch
        keeps loci in the resulting null near-independent.

    Returns
    -------
    pd.DataFrame indexed by "snv" with columns
        node, llr_null, llr_margin, score_status
    """
    log_pi_b  = np.log(np.maximum(pi_b, EPS))
    log_pi_0  = np.log(max(pi_0, EPS))
    log_1mpi0 = np.log(max(1.0 - pi_0, EPS))

    N = cna_tree.n_nodes
    L = snv_dataset.n_leaves
    node_names = np.array(
        [cna_tree.idx_to_ete[i].name for i in range(N)], dtype=object
    )

    out_snv        = []
    out_node       = []
    out_llr_null   = []
    out_llr_margin = []
    out_status     = []

    for seg, snv_idx_list in snv_dataset.snvs_by_seg.items():
        snv_idx_list = np.asarray(snv_idx_list)
        if snv_mask is not None:
            snv_idx_list = snv_idx_list[snv_mask[snv_idx_list]]
        if snv_idx_list.size == 0:
            continue

        for i0 in range(0, len(snv_idx_list), batch_size):
            batch = snv_idx_list[i0 : i0 + batch_size]

            cell_perm = None
            if cell_perm_rng is not None:
                cell_perm = cell_perm_rng.permutation(L)

            logL_nodes, logL_null = locus_loglik_batch(
                cna_tree, snv_dataset, transitions, batch,
                alpha=alpha, beta=beta, p0=p0,
                p1_fp_mode=p1_fp_mode,
                cell_perm=cell_perm,
            )

            null_score   = log_pi_0  + logL_null                          # (B,)
            branch_score = log_1mpi0 + log_pi_b[:, None] + logL_nodes     # (N, B)

            # ── MAP placement (unchanged semantics) ───────────────────────
            best_idx   = np.argmax(branch_score, axis=0)                  # (B,)
            best_score = branch_score[best_idx, np.arange(len(batch))]    # (B,)
            is_branch  = best_score > null_score      # strict: ties → "Null"
            node       = np.where(is_branch, node_names[best_idx], "Null")

            # ── Confidence quantities ─────────────────────────────────────
            n_finite = np.isfinite(branch_score).sum(axis=0)              # (B,)

            with np.errstate(invalid="ignore"):
                llr_null = best_score - null_score     # ±inf, or NaN for inf-inf

            # Second-best branch.  -inf sorts below every finite value, so the
            # overall runner-up is the finite runner-up whenever two branches
            # are finite; where fewer are, the margin is undefined rather than
            # infinite — an infinite margin is an absent comparison, not a
            # confident call.
            llr_margin = np.full(len(batch), np.nan)
            if N >= 2:
                second_score = np.partition(branch_score, N - 2, axis=0)[N - 2]
                have_margin  = n_finite >= 2
                llr_margin[have_margin] = (
                    best_score[have_margin] - second_score[have_margin]
                )

            status = np.where(
                np.isnan(llr_null), SCORE_STATUS_UNDEFINED,
                np.where(
                    n_finite == 0, SCORE_STATUS_NO_VALID_BRANCH,
                    np.where(
                        np.isneginf(null_score), SCORE_STATUS_NULL_IMPOSSIBLE,
                        np.where(
                            n_finite < 2, SCORE_STATUS_NO_MARGIN,
                            SCORE_STATUS_OK,
                        ),
                    ),
                ),
            )

            out_snv.append(snv_dataset.snv_ids[batch])
            out_node.append(node)
            out_llr_null.append(llr_null)
            out_llr_margin.append(llr_margin)
            out_status.append(status)

    if not out_snv:
        df = pd.DataFrame(columns=PLACEMENT_COLUMNS)
        df.index.name = "snv"
        return df

    df = pd.DataFrame({
        "node":         np.concatenate(out_node),
        "llr_null":     np.concatenate(out_llr_null),
        "llr_margin":   np.concatenate(out_llr_margin),
        "score_status": np.concatenate(out_status),
    }, index=pd.Index(np.concatenate(out_snv), name="snv"))

    return df


def write_placements_tsv(df, path):
    """Write a score_placements frame, preserving inf/NaN faithfully."""
    df.to_csv(path, sep="\t", float_format="%.6g", na_rep="NaN")


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

    Thin wrapper over score_placements, kept for its dict contract.

    Returns
    -------
    dict  {snv_id: node_name}   (null placements map to "Null")
    """
    df = score_placements(
        cna_tree, snv_dataset, transitions,
        alpha=alpha, beta=beta, pi_b=pi_b, pi_0=pi_0,
        p0=p0, p1_fp_mode=p1_fp_mode, batch_size=batch_size,
    )
    return df["node"].to_dict()


def run_soft_em(
    sample,
    output_root,
    config,
    input_paths,
    init_alpha=None,
    init_beta=None,
    init_pi_b=None,
    init_node_names=None,
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
    init_node_names : list or None
        Node names corresponding to init_pi_b entries.  Required when
        init_pi_b is supplied directly (not via warm_start_pkl) and the
        target tree may have a different node count (e.g. pass 2 after NNI).
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
            init_pi_b  = ws["pi_b"]
        warm_node_names = ws.get("node_names", None)
    else:
        # Node names may be supplied directly alongside init_pi_b (pipeline pass 2)
        warm_node_names = init_node_names

    init_alpha = float(init_alpha) if init_alpha is not None else config.alpha_init
    init_beta  = float(init_beta)  if init_beta  is not None else config.beta_init

    # ── Load tree / CNA / SNVs and build unified structures ────────────────
    tree_path = tree_path_override or input_paths.preprocessed_tree
    cna_tree, snv_dataset, transitions = load_structures(input_paths, tree_path)

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

    # ── Derive MAP placements + per-SNV LLRs (for refinement / downstream) ─
    print(f"[{now()}] Computing MAP placements from soft assignments...")
    placements_df = score_placements(
        cna_tree, snv_dataset, transitions,
        alpha=alpha, beta=beta,
        pi_b=pi_b, pi_0=pi_0,
        p0=config.p0,
        p1_fp_mode="one_over_c",
        batch_size=config.batch_size,
    )
    placements = placements_df["node"].to_dict()

    n_ok = int((placements_df["score_status"] == SCORE_STATUS_OK).sum())
    print(f"[{now()}] Scored {len(placements_df)} loci "
          f"({n_ok} with both LLRs finite)")

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
    # Columns snv/node keep their original semantics (including the "Null"
    # sentinel); llr_null/llr_margin/score_status are appended.
    write_placements_tsv(
        placements_df, os.path.join(sample_out, "placements_soft.tsv")
    )

    # ── Save full pickle ───────────────────────────────────────────────────
    with open(os.path.join(sample_out, "soft_em_results.pkl"), "wb") as f:
        pickle.dump({
            "pi_b":        pi_b,
            "pi_0":        pi_0,
            "node_names":  node_names,
            "alpha":       alpha,
            "beta":        beta,
            "history":     history,
            # Scoring conditions, so 'sntree score' can reproduce this fit
            "p0":          config.p0,
            "p1_fp_mode":  "one_over_c",
            "tree_path":   tree_path,
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
