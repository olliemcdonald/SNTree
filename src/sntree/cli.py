# sntree/cli.py

import argparse
import os
import pickle

from sntree.config import Config


def main():

    parser = argparse.ArgumentParser(
        prog="sntree",
        description=(
            "SNTree: Phylogeny-aware single-cell SNV inference "
            "under copy number variation.\n\n"
            "Typical usage:\n"
            "  sntree pipeline SAMPLE /path/to/input /path/to/output\n"
            "  sntree em SAMPLE /path/to/input /path/to/output\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="Available commands"
    )

    # Common arguments helper
    def add_common_args(sp):
        sp.add_argument(
            "sample",
            help="Sample name (corresponding to folder under input_root)"
        )
        sp.add_argument(
            "input_root",
            help="Root directory containing sample input data"
        )
        sp.add_argument(
            "output_root",
            help="Root directory where outputs will be written"
        )

        sp.add_argument(
            "--alpha-init",
            type=float,
            help="Initial alpha value (false positive rate)"
        )
        sp.add_argument(
            "--beta-init",
            type=float,
            help="Initial beta value (false negative/dropout rate)"
        )
        sp.add_argument(
            "--p0",
            type=float,
            help="Sequencing error baseline probability"
        )
        sp.add_argument(
            "--batch-size",
            type=int,
            help="Batch size for likelihood computation"
        )
        sp.add_argument(
            "--max-iters",
            type=int,
            help="Maximum NNI iterations for subtree refinement"
        )
        sp.add_argument(
            "--em-max-iters",
            type=int,
            help="Maximum EM iterations"
        )

    # ml
    sp_ml = subparsers.add_parser(
        "ml",
        help="Run CNA-aware maximum likelihood SNV placement"
    )
    add_common_args(sp_ml)

    # em
    sp_em = subparsers.add_parser(
        "em",
        help="Run EM estimation of alpha and beta parameters"
    )
    add_common_args(sp_em)

    # refine
    sp_refine = subparsers.add_parser(
        "refine",
        help="Refine CNA-identical subtrees using SNV likelihood"
    )
    add_common_args(sp_refine)

    # pipeline
    sp_pipeline = subparsers.add_parser(
        "pipeline",
        help="Run full pipeline (EM → subtree refinement)"
    )
    add_common_args(sp_pipeline)

    args = parser.parse_args()

    config = Config()

    # Apply overrides
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