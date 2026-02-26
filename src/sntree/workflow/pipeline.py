# sntree/workflow/pipeline.py

import time


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


from sntree.workflow.em import run_em
from sntree.workflow.refine import run_refine


def run_pipeline(sample, input_root, output_root, config):

    print(f"[{now()}] Starting full pipeline for sample {sample}")
    t0 = time.time()

    # ---- EM Stage ----
    print(f"[{now()}] --- Stage 1: EM ---")
    em_results = run_em(sample, input_root, output_root, config)

    # ---- Refinement Stage ----
    print(f"[{now()}] --- Stage 2: Refinement ---")
    refine_results = run_refine(
        sample,
        input_root,
        output_root,
        placements=em_results["placements"],
        alpha=em_results["alpha"],
        beta=em_results["beta"],
        config=config
    )

    total_runtime = time.time() - t0

    print(f"[{now()}] Full pipeline complete.")
    print(f"[{now()}] Total runtime: {total_runtime:.2f} sec")

    return refine_results