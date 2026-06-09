# sntree/workflow/pipeline.py

import os
import time

from sntree.workflow.preprocess_tree import run_preprocess
from sntree.workflow.em import run_em
from sntree.workflow.soft_em import run_soft_em
from sntree.workflow.refine import run_refine


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def run_pipeline(sample, output_root, config, input_paths):
    """
    Full SNTree pipeline.

    Default (soft-only):
        preprocess → soft EM pass 1 (joint: estimates α/β) →
        refine → soft EM pass 2 (warm-start, fixed α/β)

    With config.run_hard_em=True:
        preprocess → hard EM → soft EM pass 1 (fixed α/β) →
        refine → soft EM pass 2 (warm-start, fixed α/β)

    With config.soft_em_pass2=False:
        skip the second soft EM after refinement.
    """
    print(f"[{now()}] =======================================")
    print(f"[{now()}] Starting SNTree pipeline for {sample}")
    hard_em_str = "hard EM → " if config.run_hard_em else ""
    pass2_str   = " → soft EM pass 2" if config.soft_em_pass2 else ""
    print(f"[{now()}] Stages: preprocess → {hard_em_str}soft EM pass 1 → refine{pass2_str}")
    print(f"[{now()}] =======================================")

    t0_total = time.time()

    # ── Stage 0: Preprocess ───────────────────────────────────────────────
    print(f"[{now()}] --- Stage 0: Tree Preprocessing ---")
    t0 = time.time()
    run_preprocess(sample, output_root, input_paths)
    print(f"[{now()}] Preprocessing complete (runtime={time.time() - t0:.2f} sec)")

    # ── Stage 1a (optional): Hard EM ──────────────────────────────────────
    if config.run_hard_em:
        print(f"[{now()}] --- Stage 1a: Hard EM ---")
        t0 = time.time()
        em_results = run_em(sample, output_root, config, input_paths)
        alpha           = em_results["alpha"]
        beta            = em_results["beta"]
        hard_placements = em_results["placements"]
        print(f"[{now()}] Hard EM complete (runtime={time.time() - t0:.2f} sec)")
        joint_pass1 = False   # alpha/beta already known from hard EM
    else:
        alpha           = None
        beta            = None
        hard_placements = None
        joint_pass1     = True  # soft EM must estimate alpha/beta jointly

    # ── Stage 1b: Soft EM pass 1 ──────────────────────────────────────────
    print(f"[{now()}] --- Stage 1b: Soft EM (pass 1) ---")
    t0 = time.time()
    soft1 = run_soft_em(
        sample,
        output_root,
        config,
        input_paths,
        init_alpha=alpha,
        init_beta=beta,
        init_pi_b=None,
        tree_path_override=None,   # use MEDICC2 preprocessed tree
        output_subdir="soft_em",
        joint=joint_pass1,
        pass_label="pass 1",
    )
    alpha = soft1["alpha"]
    beta  = soft1["beta"]
    print(f"[{now()}] Soft EM pass 1 complete (runtime={time.time() - t0:.2f} sec)")

    # ── Stage 2: Refinement ───────────────────────────────────────────────
    print(f"[{now()}] --- Stage 2: Subtree Refinement ---")
    t0 = time.time()

    # Prefer hard-EM placements for refinement if available; fall back to
    # MAP placements derived from soft EM pass 1.
    refine_placements = hard_placements if hard_placements is not None else soft1["placements"]

    run_refine(
        sample,
        output_root,
        placements=refine_placements,
        alpha=alpha,
        beta=beta,
        config=config,
        input_paths=input_paths,
    )
    print(f"[{now()}] Refinement complete (runtime={time.time() - t0:.2f} sec)")

    # ── Stage 3 (optional): Soft EM pass 2 on refined tree ────────────────
    if config.soft_em_pass2:
        print(f"[{now()}] --- Stage 3: Soft EM (pass 2, warm start) ---")
        t0 = time.time()

        refined_tree_path = os.path.join(
            output_root, sample, "sntree", "refine", "refined_full_tree.new"
        )

        run_soft_em(
            sample,
            output_root,
            config,
            input_paths,
            init_alpha=alpha,
            init_beta=beta,
            init_pi_b=soft1["pi_b"],
            init_node_names=soft1["node_names"],  # needed to match pi_b to refined tree
            tree_path_override=refined_tree_path,
            output_subdir="soft_em_pass2",
            joint=False,
            pass_label="pass 2 (warm start)",
        )
        print(f"[{now()}] Soft EM pass 2 complete (runtime={time.time() - t0:.2f} sec)")

    total_runtime = time.time() - t0_total
    print(f"[{now()}] =======================================")
    print(f"[{now()}] Pipeline finished. Total runtime: {total_runtime:.2f} sec")
    print(f"[{now()}] =======================================")
