import os
from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class InputPaths:
    medicc_tree: str
    cna_profiles: str
    cna_distances: str
    sample_mapping: str
    vcf: str
    normal_name: str
    preprocessed_tree: str


REQUIRED_BY_COMMAND = {
    "preprocess": ("medicc_tree", "cna_distances"),
    "ml": ("preprocessed_tree", "sample_mapping", "cna_profiles", "vcf"),
    "em": ("preprocessed_tree", "sample_mapping", "cna_profiles", "vcf"),
    "refine": (
        "preprocessed_tree",
        "sample_mapping",
        "cna_profiles",
        "cna_distances",
        "vcf",
    ),
    "pipeline": ("medicc_tree", "cna_distances", "sample_mapping", "cna_profiles", "vcf"),
}


def default_input_paths(sample: str, input_root: str, output_root: str) -> InputPaths:
    sample_base = os.path.join(output_root, sample, "sntree")

    return InputPaths(
        medicc_tree=os.path.join(
            input_root,
            sample,
            "medicc2",
            f"{sample}_final_tree.new",
        ),
        cna_profiles=os.path.join(
            input_root,
            sample,
            "medicc2",
            f"{sample}_final_cn_profiles.tsv",
        ),
        cna_distances=os.path.join(
            input_root,
            sample,
            "medicc2",
            f"{sample}_pairwise_distances.tsv",
        ),
        sample_mapping=os.path.join(input_root, sample, "chisel", f"{sample}.info.tsv"),
        vcf=os.path.join(input_root, sample, "snv", "consensus_singlecell_counts.vcf.gz"),
        normal_name=os.path.join(
            input_root,
            sample,
            "normal_cells",
            f"{sample}_normal_markdup.bam",
        ),
        preprocessed_tree=os.path.join(sample_base, "tree_preprocessed.new"),
    )


def resolve_input_paths(args, command: str) -> InputPaths:
    paths = default_input_paths(args.sample, args.input_root, args.output_root)

    overrides = {
        "medicc_tree": args.medicc_tree,
        "cna_profiles": args.cna_profiles,
        "cna_distances": args.cna_distances,
        "sample_mapping": args.sample_mapping,
        "vcf": args.vcf,
        "normal_name": args.normal_name,
        "preprocessed_tree": args.preprocessed_tree,
    }

    for field, value in overrides.items():
        if value is not None:
            setattr(paths, field, value)

    validate_input_paths(paths, command)
    return paths


def required_fields(command: str) -> Iterable[str]:
    try:
        return REQUIRED_BY_COMMAND[command]
    except KeyError as exc:
        raise ValueError(f"Unknown SNTree command: {command}") from exc


def validate_input_paths(paths: InputPaths, command: str) -> None:
    missing: List[str] = []

    for field in required_fields(command):
        path = getattr(paths, field)
        if not os.path.exists(path):
            missing.append(f"{field}: {path}")

    if missing:
        details = "\n  ".join(missing)
        raise FileNotFoundError(
            f"Missing required input file(s) for '{command}':\n  {details}"
        )
