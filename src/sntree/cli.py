# sntree/cli.py

import argparse
import os
import pickle

from sntree.config import Config


def main():

    parser = argparse.ArgumentParser(prog="sntree")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for cmd in ["ml", "em", "refine", "pipeline"]:
        sp = subparsers.add_parser(cmd)
        sp.add_argument("sample")
        sp.add_argument("input_root")
        sp.add_argument("output_root")

        sp.add_argument("--alpha-init", type=float)
        sp.add_argument("--beta-init", type=float)
        sp.add_argument("--p0", type=float)
        sp.add_argument("--batch-size", type=int)
        sp.add_argument("--max-iters", type=int)
        sp.add_argument("--em-max-iters", type=int)

    args = parser.parse_args()

    config = Config()

    if args.alpha_init is not None:
        config.alpha_init = args.alpha_init
    if args.beta_init is not None:
        config.beta_init = args.beta_init
    if args.p0 is not None:
        config.p0 = args.p0
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.max_iters is not None:
        config.max_iters = args.max_iters
    if args.em_max_iters is not None:
        config.em_max_iter = args.em_max_iters

    if args.command == "ml":
        from sntree.workflow.ml import run_ml
        run_ml(args.sample, args.input_root, args.output_root, config)

    elif args.command == "em":
        from sntree.workflow.em import run_em
        run_em(args.sample, args.input_root, args.output_root, config)

    elif args.command == "refine":
        from sntree.workflow.refine import run_refine

        sample_out = os.path.join(args.output_root, args.sample)

        # Prefer EM results
        em_path = os.path.join(sample_out, "em", "em_results.pkl")
        ml_results_path = os.path.join(sample_out, "ml", "ml_results.pkl")
        ml_legacy_path = os.path.join(sample_out, "ml", "placements_ml.pkl")

        if os.path.exists(em_path):
            with open(em_path, "rb") as f:
                em_results = pickle.load(f)

            placements = em_results["placements"]
            alpha = em_results["alpha"]
            beta = em_results["beta"]

        elif os.path.exists(ml_results_path):
            with open(ml_results_path, "rb") as f:
                ml_results = pickle.load(f)

            placements = ml_results["placements"]
            alpha = ml_results.get("alpha", config.alpha_init)
            beta = ml_results.get("beta", config.beta_init)
        elif os.path.exists(ml_legacy_path):
            with open(ml_legacy_path, "rb") as f:
                placements = pickle.load(f)

            alpha = config.alpha_init
            beta = config.beta_init

        else:
            raise RuntimeError(
                "No EM or ML results found. Run 'sntree em' or 'sntree ml' first."
            )

        run_refine(
            args.sample,
            args.input_root,
            args.output_root,
            placements,
            alpha,
            beta,
            config
        )

    elif args.command == "pipeline":
        from sntree.workflow.pipeline import run_pipeline
        run_pipeline(args.sample, args.input_root, args.output_root, config)
