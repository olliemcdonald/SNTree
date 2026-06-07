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
            "Typical usage (soft EM, default):\n"
            "  sntree pipeline SAMPLE /input /output\n\n"
            "Staged workflow:\n"
            "  sntree preprocess SAMPLE /input /output\n"
            "  sntree soft-em   SAMPLE /input /output\n"
            "  sntree refine    SAMPLE /input /output\n"
            "  sntree soft-em   SAMPLE /input /output \\\n"
            "      --refined-tree /output/SAMPLE/sntree/refine/refined_full_tree.new \\\n"
            "      --warm-start-pkl /output/SAMPLE/sntree/soft_em/soft_em_results.pkl\n\n"
            "With hard EM:\n"
            "  sntree pipeline SAMPLE /input /output --hard-em\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="Available commands",
    )

    # ── Common path / tuning arguments ────────────────────────────────────

    def add_common_args(sp):
        sp.add_argument("sample",      help="Sample name")
        sp.add_argument("input_root",  help="Root directory containing sample input data")
        sp.add_argument("output_root", help="Root directory where outputs will be written")

        g = sp.add_argument_group("input path overrides")
        g.add_argument("--medicc-tree",       help="MEDICC2 Newick tree path")
        g.add_argument("--cna-profiles",      help="MEDICC2 final CN profiles TSV")
        g.add_argument("--cna-distances",     help="MEDICC2 pairwise distances TSV")
        g.add_argument("--sample-mapping",    help="CHISEL sample info TSV")
        g.add_argument("--vcf",               help="SNV VCF path")
        g.add_argument("--normal-name",       help="Normal sample name in VCF")
        g.add_argument("--preprocessed-tree", help="Preprocessed tree path")

        g2 = sp.add_argument_group("EM tuning")
        g2.add_argument("--alpha-init",        type=float, help="Initial alpha (FP rate)")
        g2.add_argument("--beta-init",         type=float, help="Initial beta (FN rate)")
        g2.add_argument("--p0",                type=float, help="Background error probability")
        g2.add_argument("--pi0",               type=float, help="Prior null-placement probability")
        g2.add_argument("--batch-size",        type=int,   help="Batch size for likelihood DP")
        g2.add_argument("--nni-max-iters",     type=int,   help="Max NNI iterations")
        g2.add_argument("--em-max-iters",      type=int,   help="Max hard EM iterations")
        g2.add_argument("--soft-em-max-iters", type=int,   help="Max soft EM iterations (default 50)")
        g2.add_argument("--alpha-dir",         type=float,
                        help="Dirichlet concentration on pi_b (default 1.0)")

    # ── preprocess ────────────────────────────────────────────────────────
    sp_pre = subparsers.add_parser(
        "preprocess",
        help="Preprocess MEDICC2 tree (normalise + split CNA-identical clades)",
    )
    add_common_args(sp_pre)

    # ── ml ────────────────────────────────────────────────────────────────
    sp_ml = subparsers.add_parser(
        "ml",
        help="CNA-aware maximum likelihood SNV placement",
    )
    add_common_args(sp_ml)

    # ── em ────────────────────────────────────────────────────────────────
    sp_em = subparsers.add_parser(
        "em",
        help="Hard EM: estimate alpha/beta and produce MAP SNV placements",
    )
    add_common_args(sp_em)

    # ── soft-em ───────────────────────────────────────────────────────────
    sp_sem = subparsers.add_parser(
        "soft-em",
        help=(
            "Soft branch-proportion EM: infer relative mutation burdens pi_b.\n"
            "Run once on the MEDICC2 tree, then again after 'sntree refine'\n"
            "using --refined-tree and --warm-start-pkl for a warm-started pass."
        ),
    )
    add_common_args(sp_sem)
    sp_sem.add_argument(
        "--refined-tree",
        default=None,
        help="Use this Newick tree instead of the preprocessed MEDICC2 tree.\n"
             "Typically the output of 'sntree refine' "
             "(.../sntree/refine/refined_full_tree.new).",
    )
    sp_sem.add_argument(
        "--warm-start-pkl",
        default=None,
        help="Path to a previous soft_em_results.pkl.\n"
             "pi_b and alpha/beta are loaded from it to warm-start this run.\n"
             "Typically used for pass 2 after refinement.",
    )
    sp_sem.add_argument(
        "--soft-em-joint",
        action="store_true",
        default=False,
        help="Jointly update alpha/beta during soft EM\n"
             "(default: auto — on when no hard EM results exist, off otherwise).",
    )
    sp_sem.add_argument(
        "--output-subdir",
        default=None,
        help="Output subdirectory name under .../sntree/ (default: soft_em or soft_em_pass2).",
    )

    # ── refine ────────────────────────────────────────────────────────────
    sp_refine = subparsers.add_parser(
        "refine",
        help="Refine CNA-identical subtrees using SNV likelihoods (NNI)",
    )
    add_common_args(sp_refine)

    # ── pipeline ──────────────────────────────────────────────────────────
    sp_pipeline = subparsers.add_parser(
        "pipeline",
        help=(
            "Full pipeline.\n"
            "Default: soft EM pass 1 → refine → soft EM pass 2 (warm start).\n"
            "Use --hard-em to add a hard EM stage before the soft passes."
        ),
    )
    add_common_args(sp_pipeline)
    sp_pipeline.add_argument(
        "--hard-em",
        action="store_true",
        default=False,
        help="Run hard EM (alpha/beta estimation + MAP placements) before\n"
             "the soft EM passes.  By default the soft EM estimates alpha/beta\n"
             "jointly in pass 1.",
    )
    sp_pipeline.add_argument(
        "--no-soft-em-pass2",
        action="store_true",
        default=False,
        help="Skip the second soft EM after refinement.",
    )
    sp_pipeline.add_argument(
        "--no-refine",
        action="store_true",
        default=False,
        help="Skip refinement and pass 2 entirely (soft EM pass 1 only).",
    )

    args = parser.parse_args()

    # ── Build config ───────────────────────────────────────────────────────
    config = Config()

    if getattr(args, "alpha_init",        None) is not None: config.alpha_init        = args.alpha_init
    if getattr(args, "beta_init",         None) is not None: config.beta_init         = args.beta_init
    if getattr(args, "p0",                None) is not None: config.p0                = args.p0
    if getattr(args, "pi0",               None) is not None: config.pi0               = args.pi0
    if getattr(args, "batch_size",        None) is not None: config.batch_size        = args.batch_size
    if getattr(args, "nni_max_iters",     None) is not None: config.nni_max_iters     = args.nni_max_iters
    if getattr(args, "em_max_iters",      None) is not None: config.em_max_iter       = args.em_max_iters
    if getattr(args, "soft_em_max_iters", None) is not None: config.soft_em_max_iter  = args.soft_em_max_iters
    if getattr(args, "alpha_dir",         None) is not None: config.alpha_dir         = args.alpha_dir

    # Pipeline-specific flags
    if getattr(args, "hard_em",           False): config.run_hard_em   = True
    if getattr(args, "no_soft_em_pass2",  False): config.soft_em_pass2 = False
    if getattr(args, "no_refine",         False):
        config.soft_em_pass2 = False   # no refine → no pass 2

    from sntree.io.input_paths import resolve_input_paths
    input_paths = resolve_input_paths(args, args.command)

    # ── Dispatch ───────────────────────────────────────────────────────────

    if args.command == "preprocess":
        from sntree.workflow.preprocess_tree import run_preprocess
        run_preprocess(args.sample, args.output_root, input_paths)

    elif args.command == "ml":
        from sntree.workflow.ml import run_ml
        run_ml(args.sample, args.output_root, config, input_paths)

    elif args.command == "em":
        from sntree.workflow.em import run_em
        run_em(args.sample, args.output_root, config, input_paths)

    elif args.command == "soft-em":
        from sntree.workflow.soft_em import run_soft_em

        joint_override = args.soft_em_joint if args.soft_em_joint else None

        # Infer default output subdir from whether a warm-start pkl is given
        if args.output_subdir is not None:
            subdir = args.output_subdir
        elif args.warm_start_pkl is not None or args.refined_tree is not None:
            subdir = "soft_em_pass2"
        else:
            subdir = "soft_em"

        run_soft_em(
            args.sample,
            args.output_root,
            config,
            input_paths,
            tree_path_override=args.refined_tree,
            warm_start_pkl=args.warm_start_pkl,
            output_subdir=subdir,
            joint=joint_override,
        )

    elif args.command == "refine":
        from sntree.workflow.refine import run_refine

        sample_base = os.path.join(args.output_root, args.sample, "sntree")

        # Prefer soft EM pass 1 results; fall back to hard EM; then error
        soft_pkl  = os.path.join(sample_base, "soft_em",  "soft_em_results.pkl")
        soft_place = os.path.join(sample_base, "soft_em", "placements_soft.tsv")
        em_pkl    = os.path.join(sample_base, "em",       "em_results.pkl")

        if os.path.exists(soft_pkl):
            with open(soft_pkl, "rb") as f:
                soft_res = pickle.load(f)
            alpha = float(soft_res["alpha"])
            beta  = float(soft_res["beta"])
            import pandas as pd
            placements = pd.read_csv(soft_place, sep="\t", index_col="snv")["node"].to_dict()

        elif os.path.exists(em_pkl):
            with open(em_pkl, "rb") as f:
                em_res = pickle.load(f)
            placements = em_res["placements"]
            alpha      = em_res["alpha"]
            beta       = em_res["beta"]

        else:
            raise RuntimeError(
                "No soft EM or hard EM results found.\n"
                "Run 'sntree soft-em' or 'sntree em' first."
            )

        run_refine(
            args.sample,
            args.output_root,
            placements,
            alpha,
            beta,
            config,
            input_paths,
        )

    elif args.command == "pipeline":
        from sntree.workflow.pipeline import run_pipeline

        if getattr(args, "no_refine", False):
            # Soft EM pass 1 only — run it directly
            from sntree.workflow.soft_em import run_soft_em
            run_soft_em(
                args.sample,
                args.output_root,
                config,
                input_paths,
                joint=None,   # auto-detect
                output_subdir="soft_em",
                pass_label="pass 1 (no refine)",
            )
        else:
            run_pipeline(args.sample, args.output_root, config, input_paths)
