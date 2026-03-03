# sntree/workflow/pipeline.py

import time

from sntree.workflow.preprocess_tree import run_preprocess
from sntree.workflow.em import run_em
from sntree.workflow.refine import run_refine


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def run_pipeline(sample, input_root, output_root, config):

    print(f"[{now()}] =======================================")
    print(f"[{now()}] Starting full SNTree pipeline for {sample}")
    print(f"[{now()}] =======================================")

    t0_total = time.time()

    # -------------------------------------------------
    # Stage 0: Preprocess tree
    # -------------------------------------------------
    print(f"[{now()}] --- Stage 0: Tree Preprocessing ---")
    t0 = time.time()

    run_preprocess(sample, input_root, output_root)

    print(f"[{now()}] Preprocessing complete "
          f"(runtime={time.time() - t0:.2f} sec)")

    # -------------------------------------------------
    # Stage 1: EM parameter estimation
    # -------------------------------------------------
    print(f"[{now()}] --- Stage 1: EM Estimation ---")
    t0 = time.time()

    em_results = run_em(sample, input_root, output_root, config)

    print(f"[{now()}] EM stage complete "
          f"(runtime={time.time() - t0:.2f} sec)")

    # -------------------------------------------------
    # Stage 2: Subtree refinement
    # -------------------------------------------------
    print(f"[{now()}] --- Stage 2: Subtree Refinement ---")
    t0 = time.time()

    refine_results = run_refine(
        sample,
        input_root,
        output_root,
        placements=em_results["placements"],
        alpha=em_results["alpha"],
        beta=em_results["beta"],
        config=config
    )

    print(f"[{now()}] Refinement stage complete "
          f"(runtime={time.time() - t0:.2f} sec)")

    total_runtime = time.time() - t0_total

    print(f"[{now()}] =======================================")
    print(f"[{now()}] SNTree pipeline finished successfully")
    print(f"[{now()}] Total runtime: {total_runtime:.2f} sec")
    print(f"[{now()}] =======================================")

    return refine_results