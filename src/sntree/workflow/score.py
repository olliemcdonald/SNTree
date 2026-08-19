# sntree/workflow/score.py

import os
import time
import pickle
import numpy as np
import pandas as pd

from sntree.workflow.load_inputs import load_structures
from sntree.workflow.soft_em import (
    score_placements,
    write_placements_tsv,
    SCORE_STATUS_OK,
)

#: Quantiles of the permuted llr_null reported as candidate thresholds.
NULL_QUANTILES = [0.5, 0.9, 0.95, 0.99, 0.999, 0.9999]


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _load_results(results_pkl):
    with open(results_pkl, "rb") as f:
        res = pickle.load(f)

    missing = [k for k in ("pi_b", "pi_0", "node_names", "alpha", "beta")
               if k not in res]
    if missing:
        raise RuntimeError(
            f"{results_pkl} is missing required key(s): {', '.join(missing)}. "
            "Expected a soft_em_results.pkl written by 'sntree soft-em'."
        )
    return res


def _match_pi_b(pi_b, pkl_node_names, tree_node_names, tree_path):
    """
    Reorder pi_b from the pkl's node ordering into the loaded tree's ordering.

    Strict by design: unlike the warm-start path in run_soft_em, filling an
    unmatched node with a mean is meaningless when scoring — a mismatch means
    the tree is not the one the parameters were fit on, and every LLR would be
    computed under the wrong prior.
    """
    name_to_pi = dict(zip(pkl_node_names, pi_b))
    missing = [n for n in tree_node_names if n not in name_to_pi]

    if missing:
        raise RuntimeError(
            f"{len(missing)} of {len(tree_node_names)} nodes in {tree_path} "
            f"are absent from the fitted parameters "
            f"(e.g. {', '.join(map(str, missing[:5]))}).\n"
            "The scoring pass must use the same tree the parameters were fit "
            "on — pass --refined-tree when scoring a pass-2 fit."
        )

    return np.array([name_to_pi[n] for n in tree_node_names], dtype=np.float64)


def _subsample_mask(n_snvs, subsample_loci, subsample_seed):
    """
    Boolean mask over loci.  Seeded separately from the permutation seed so
    that the observed pass and every permutation replicate score exactly the
    same loci.
    """
    if subsample_loci is None or subsample_loci >= n_snvs:
        return None

    rng = np.random.default_rng(subsample_seed)
    chosen = rng.choice(n_snvs, size=int(subsample_loci), replace=False)
    mask = np.zeros(n_snvs, dtype=bool)
    mask[chosen] = True
    return mask


def _status_counts(df, label):
    counts = df["score_status"].value_counts()
    row = {"run": label, "n_loci": len(df)}
    row.update({f"n_{k}": int(v) for k, v in counts.items()})
    return row


def placement_layer(nodes, root_name):
    """
    Split placements into the layers the permutation null behaves differently
    on: "truncal" (the root — every cell is a descendant), "subclonal" (any
    other branch) and "null".
    """
    return np.where(
        nodes == "Null", "null",
        np.where(nodes == root_name, "truncal", "subclonal"),
    )


def _quantiles(values, quantiles):
    if values.size == 0:
        return np.full(len(quantiles), np.nan)
    return np.array([float(np.quantile(values, q)) for q in quantiles])


def summarise_llr(observed_df, permuted_dfs, root_name=None):
    """
    Quantiles of llr_null observed against the permutation null, plus the
    number of observed loci that survive each permuted quantile as a
    threshold — reported per placement layer as well as pooled.

    Stratifying by layer is not cosmetic.  A truncal locus is very nearly
    *invariant* under tip permutation: the root's clade is every cell, so
    shuffling which cell sits at which tip leaves its llr_null essentially
    unchanged (only the per-tip CN entering p1 moves at all).  Only subclonal
    placements lose concordance.  When the truncal layer dominates the data
    (as it does at low coverage), a pooled null's tail consists entirely of
    loci the permutation could not touch, and the resulting threshold is far
    too high for the subclonal layer it is meant to filter.  Threshold the
    subclonal layer against the subclonal rows of this table.

    Only "ok" loci enter the quantiles, in both the observed and the permuted
    runs — mixing in loci whose llr_null is infinite or undefined would make
    the two sets non-comparable and the threshold wrong.

    Layers are assigned from the *observed* placement, so a permuted locus is
    compared against the layer it was actually placed in.
    """
    obs = observed_df
    obs_layer = (
        placement_layer(obs["node"].to_numpy(), root_name)
        if root_name is not None
        else np.full(len(obs), "all", dtype=object)
    )
    obs_ok = (obs["score_status"] == SCORE_STATUS_OK).to_numpy()

    layers = ["all"]
    if root_name is not None:
        layers += [l for l in ("truncal", "subclonal", "null")
                   if (obs_layer == l).any()]

    frames = []
    for layer in layers:
        in_layer = np.ones(len(obs), dtype=bool) if layer == "all" else (obs_layer == layer)
        layer_ids = obs.index[in_layer]

        obs_vals = obs.loc[in_layer & obs_ok, "llr_null"].to_numpy()
        block = pd.DataFrame({
            "layer":        layer,
            "quantile":     NULL_QUANTILES,
            "n_observed_ok": len(obs_vals),
            "observed_llr": _quantiles(obs_vals, NULL_QUANTILES),
        })

        for label, df in permuted_dfs.items():
            sub = df.loc[df.index.intersection(layer_ids)]
            perm_vals = sub.loc[
                sub["score_status"] == SCORE_STATUS_OK, "llr_null"
            ].to_numpy()

            thresholds = _quantiles(perm_vals, NULL_QUANTILES)
            block[f"{label}_llr"] = thresholds
            block[f"{label}_n_obs_above"] = [
                int((obs_vals > thr).sum()) if np.isfinite(thr) else 0
                for thr in thresholds
            ]
            block[f"{label}_frac_obs_above"] = [
                float((obs_vals > thr).mean())
                if (np.isfinite(thr) and obs_vals.size) else np.nan
                for thr in thresholds
            ]

        frames.append(block)

    return pd.concat(frames, ignore_index=True)


def run_score(
    sample,
    output_root,
    config,
    input_paths,
    results_pkl=None,
    tree_path_override=None,
    output_subdir=None,
    permute=False,
    seed=0,
    replicates=1,
    subsample_loci=None,
    subsample_seed=0,
):
    """
    Score-only pass over converged soft EM parameters, with an optional
    permutation null for llr_null.

    Loads parameters from an existing soft_em_results.pkl and runs one E-step
    pass rather than re-running EM.  With permute=True, each locus batch is
    scored against a fresh shuffle of the cell → tip assignment (read counts
    per cell left intact), which destroys clade concordance for real mutations
    while leaving scattered artefacts unaffected — an empirical null for
    llr_null.

    Returns
    -------
    dict with keys: observed (DataFrame), permuted (dict label → DataFrame),
                    summary (DataFrame or None), output_dir (str)
    """
    sample_base = os.path.join(output_root, sample, "sntree")

    if results_pkl is None:
        results_pkl = os.path.join(sample_base, "soft_em", "soft_em_results.pkl")
    if not os.path.exists(results_pkl):
        raise RuntimeError(
            f"Soft EM results not found at {results_pkl}. "
            "Run 'sntree soft-em' first, or pass --results-pkl."
        )

    # Outputs land alongside the parameters they were scored from
    if output_subdir is None:
        sample_out = os.path.dirname(os.path.abspath(results_pkl))
    else:
        sample_out = os.path.join(sample_base, output_subdir)
    os.makedirs(sample_out, exist_ok=True)

    print(f"[{now()}] ========================================")
    print(f"[{now()}] Placement scoring for {sample}")
    print(f"[{now()}] ========================================")
    t0 = time.time()

    # ── Converged parameters ───────────────────────────────────────────────
    print(f"[{now()}] Loading parameters from {results_pkl}")
    res = _load_results(results_pkl)
    alpha      = float(res["alpha"])
    beta       = float(res["beta"])
    pi_0       = float(res["pi_0"])
    p0         = float(res.get("p0", config.p0))
    p1_fp_mode = res.get("p1_fp_mode", "one_over_c")
    print(f"[{now()}] alpha={alpha:.5f}  beta={beta:.5f}  "
          f"pi_0={pi_0:.4f}  p0={p0:.5f}")

    # ── Data structures ────────────────────────────────────────────────────
    tree_path = (
        tree_path_override
        or res.get("tree_path")
        or input_paths.preprocessed_tree
    )
    cna_tree, snv_dataset, transitions = load_structures(input_paths, tree_path)

    tree_node_names = [
        cna_tree.idx_to_ete[i].name for i in range(cna_tree.n_nodes)
    ]
    pi_b = _match_pi_b(res["pi_b"], res["node_names"], tree_node_names, tree_path)

    # ── Locus subsample ────────────────────────────────────────────────────
    snv_mask = _subsample_mask(snv_dataset.n_snvs, subsample_loci, subsample_seed)
    n_scored = snv_dataset.n_snvs if snv_mask is None else int(snv_mask.sum())
    print(f"[{now()}] Scoring {n_scored} of {snv_dataset.n_snvs} loci"
          + ("" if snv_mask is None else f" (subsample seed {subsample_seed})"))

    score_kwargs = dict(
        alpha=alpha, beta=beta, pi_b=pi_b, pi_0=pi_0,
        p0=p0, p1_fp_mode=p1_fp_mode,
        batch_size=config.batch_size,
        snv_mask=snv_mask,
    )

    # ── Observed pass ──────────────────────────────────────────────────────
    print(f"[{now()}] Observed pass (one E-step)...")
    t_obs = time.time()
    observed = score_placements(
        cna_tree, snv_dataset, transitions, **score_kwargs
    )
    write_placements_tsv(observed, os.path.join(sample_out, "llr_observed.tsv"))
    print(f"[{now()}] Observed pass complete "
          f"(runtime={time.time() - t_obs:.2f} sec)")

    status_rows = [_status_counts(observed, "observed")]

    # ── Permutation null ───────────────────────────────────────────────────
    permuted = {}
    if permute:
        for r in range(int(replicates)):
            rep_seed = int(seed) + r
            label = f"permuted_seed{rep_seed}"
            print(f"[{now()}] Permuted pass {r + 1}/{replicates} "
                  f"(seed {rep_seed})...")
            t_perm = time.time()

            # Independent stream from the subsample RNG: the loci stay fixed
            # across replicates, only the shuffle changes.
            perm_rng = np.random.default_rng([rep_seed, 1])
            df = score_placements(
                cna_tree, snv_dataset, transitions,
                cell_perm_rng=perm_rng, **score_kwargs
            )
            write_placements_tsv(df, os.path.join(sample_out, f"llr_{label}.tsv"))
            permuted[label] = df
            status_rows.append(_status_counts(df, label))
            print(f"[{now()}] Permuted pass {r + 1} complete "
                  f"(runtime={time.time() - t_perm:.2f} sec)")

    # ── Summary ────────────────────────────────────────────────────────────
    status_df = pd.DataFrame(status_rows).fillna(0)
    status_df.to_csv(
        os.path.join(sample_out, "llr_status_counts.tsv"), sep="\t", index=False
    )

    summary = None
    if permuted:
        root_name = cna_tree.idx_to_ete[cna_tree.preorder[0]].name
        summary = summarise_llr(observed, permuted, root_name=root_name)
        summary.to_csv(
            os.path.join(sample_out, "llr_summary.tsv"),
            sep="\t", index=False, float_format="%.6g", na_rep="NaN",
        )
        print(f"[{now()}] llr_null quantiles (status=='{SCORE_STATUS_OK}' loci only):")
        print(summary.to_string(index=False))
        print(f"[{now()}] NOTE: truncal placements barely move under tip "
              f"permutation (the root's clade is every cell), so threshold the")
        print(f"[{now()}]       subclonal layer against the 'subclonal' rows, "
              f"not the pooled 'all' rows.")

    print(f"[{now()}] Results written to {sample_out}/")
    print(f"[{now()}] Scoring complete (runtime={time.time() - t0:.2f} sec)")

    return {
        "observed":   observed,
        "permuted":   permuted,
        "summary":    summary,
        "output_dir": sample_out,
    }
