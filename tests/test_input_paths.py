from argparse import Namespace

import pytest

from sntree.io.input_paths import resolve_input_paths


def _args(tmp_path, **overrides):
    values = {
        "sample": "S1",
        "input_root": str(tmp_path / "input"),
        "output_root": str(tmp_path / "output"),
        "medicc_tree": None,
        "cna_profiles": None,
        "cna_distances": None,
        "sample_mapping": None,
        "vcf": None,
        "normal_name": None,
        "preprocessed_tree": None,
    }
    values.update(overrides)
    return Namespace(**values)


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def test_default_preprocess_paths(tmp_path):
    input_root = tmp_path / "input"
    _touch(input_root / "S1" / "medicc2" / "S1_final_tree.new")
    _touch(input_root / "S1" / "medicc2" / "S1_pairwise_distances.tsv")

    paths = resolve_input_paths(_args(tmp_path), "preprocess")

    assert paths.medicc_tree == str(input_root / "S1" / "medicc2" / "S1_final_tree.new")
    assert paths.cna_distances == str(
        input_root / "S1" / "medicc2" / "S1_pairwise_distances.tsv"
    )


def test_overrides_replace_default_paths(tmp_path):
    custom_tree = tmp_path / "custom" / "tree.new"
    custom_dist = tmp_path / "custom" / "dist.tsv"
    _touch(custom_tree)
    _touch(custom_dist)

    paths = resolve_input_paths(
        _args(
            tmp_path,
            medicc_tree=str(custom_tree),
            cna_distances=str(custom_dist),
        ),
        "preprocess",
    )

    assert paths.medicc_tree == str(custom_tree)
    assert paths.cna_distances == str(custom_dist)


def test_missing_required_file_reports_command_and_field(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        resolve_input_paths(_args(tmp_path), "em")

    message = str(exc.value)
    assert "em" in message
    assert "preprocessed_tree" in message
    assert "sample_mapping" in message
    assert "cna_profiles" in message
    assert "vcf" in message
