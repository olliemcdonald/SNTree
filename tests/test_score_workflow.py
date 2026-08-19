"""
Tests for the standalone scoring pass (sntree score): parameter reloading,
locus subsampling, the permutation null and the threshold summary.

run_score is exercised end to end with only the tree/CNA/VCF loading stubbed
out, so the pkl handling, pi_b re-matching, output files and summary are the
real ones.
"""

import os
import pickle

import numpy as np
import pandas as pd
import pytest

from sntree.config import Config
from sntree.workflow import score as score_mod
from sntree.workflow.score import (
    run_score,
    summarise_llr,
    _match_pi_b,
    _subsample_mask,
    NULL_QUANTILES,
)
from sntree.workflow.soft_em import (
    SCORE_STATUS_OK,
    SCORE_STATUS_NO_VALID_BRANCH,
)

from test_placement_scores import (
    CELLS, CLADE_CELLS, SCATTERED_CELLS, SISTER_CELLS,
    ALPHA, BETA, P0, _toy_dataset,
)


# ── pi_b re-matching ──────────────────────────────────────────────────────

def test_match_pi_b_reorders_by_node_name():
    pi_b = np.array([0.1, 0.2, 0.7])
    out = _match_pi_b(pi_b, ["a", "b", "c"], ["c", "a", "b"], "tree.new")

    np.testing.assert_allclose(out, [0.7, 0.1, 0.2])


def test_match_pi_b_refuses_a_tree_the_parameters_were_not_fit_on():
    """
    Silently mean-filling an unmatched node (as the warm-start path does) would
    score every locus under the wrong prior.
    """
    with pytest.raises(RuntimeError, match="--refined-tree"):
        _match_pi_b(np.array([0.5, 0.5]), ["a", "b"], ["a", "b", "new"], "t.new")


# ── locus subsampling ─────────────────────────────────────────────────────

def test_subsample_mask_is_stable_across_calls():
    """The observed pass and every replicate must score the same loci."""
    a = _subsample_mask(1000, 100, subsample_seed=7)
    b = _subsample_mask(1000, 100, subsample_seed=7)

    assert a.sum() == 100
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, _subsample_mask(1000, 100, subsample_seed=8))


def test_subsample_mask_is_none_when_not_subsampling():
    assert _subsample_mask(100, None, 0) is None
    assert _subsample_mask(100, 500, 0) is None


# ── summary ───────────────────────────────────────────────────────────────

def _frame(llr, status=None, node=None):
    n = len(llr)
    return pd.DataFrame({
        "node": node or ["b1"] * n,
        "llr_null": llr,
        "llr_margin": [1.0] * n,
        "score_status": status or [SCORE_STATUS_OK] * n,
    }, index=[f"s{i}" for i in range(n)])


def test_summary_quantiles_ignore_non_ok_loci():
    """
    An infinite llr_null from a degenerate locus must not be allowed to drag a
    threshold quantile — non-ok loci are excluded from both distributions.
    """
    observed = _frame(
        [1.0, 2.0, 3.0, np.inf],
        [SCORE_STATUS_OK] * 3 + [SCORE_STATUS_NO_VALID_BRANCH],
    )
    permuted = {"permuted_seed0": _frame([0.0, 0.5, 1.0, 2.0])}

    summary = summarise_llr(observed, permuted)

    assert np.isfinite(summary["observed_llr"]).all()
    assert summary["observed_llr"].max() <= 3.0
    assert (summary["n_observed_ok"] == 3).all()


def test_summary_counts_observed_loci_above_each_permuted_quantile():
    observed = _frame([0.0, 5.0, 10.0, 20.0])
    permuted = {"permuted_seed0": _frame([0.0, 1.0, 2.0, 3.0])}

    summary = summarise_llr(observed, permuted)
    row = summary[summary["quantile"] == 0.5].iloc[0]

    # median of the permuted null is 1.5; three observed loci exceed it
    assert row["layer"] == "all"
    assert row["permuted_seed0_llr"] == pytest.approx(1.5)
    assert row["permuted_seed0_n_obs_above"] == 3
    assert row["permuted_seed0_frac_obs_above"] == pytest.approx(0.75)


def test_summary_stratifies_by_placement_layer():
    """
    Truncal placements are invariant under tip permutation, so pooling them
    into the null puts an untouchable tail on it.  The subclonal rows must
    give a threshold derived only from subclonal loci.
    """
    nodes = ["root", "root", "b1", "b1", "Null"]
    observed = _frame([900.0, 910.0, 10.0, 20.0, -5.0], node=nodes)
    # Permutation leaves the truncal loci alone, collapses the subclonal ones
    permuted = {"permuted_seed0": _frame([900.0, 910.0, 1.0, 2.0, -5.0], node=nodes)}

    summary = summarise_llr(observed, permuted, root_name="root")

    assert set(summary["layer"]) == {"all", "truncal", "subclonal", "null"}

    sub = summary[(summary.layer == "subclonal") & (summary["quantile"] == 0.5)].iloc[0]
    pooled = summary[(summary.layer == "all") & (summary["quantile"] == 0.5)].iloc[0]

    assert sub["n_observed_ok"] == 2
    assert sub["permuted_seed0_llr"] == pytest.approx(1.5)
    assert sub["permuted_seed0_n_obs_above"] == 2      # both subclonal loci survive

    # The pooled threshold is dragged up by the permutation-invariant truncal
    # loci, which is exactly why the subclonal layer must not use it.
    assert pooled["permuted_seed0_llr"] > sub["permuted_seed0_llr"]

    trunc = summary[(summary.layer == "truncal") & (summary["quantile"] == 0.5)].iloc[0]
    assert trunc["permuted_seed0_llr"] == pytest.approx(trunc["observed_llr"])


# ── run_score end to end ──────────────────────────────────────────────────

@pytest.fixture
def scoring_run(tmp_path, monkeypatch):
    """A sample directory with a soft_em_results.pkl and stubbed data loading."""
    loci = {
        "clade":     CLADE_CELLS,
        "scattered": SCATTERED_CELLS,
        "sister":    SISTER_CELLS,
        "empty":     [],
    }
    cna_tree, snv_dataset, transitions = _toy_dataset(loci)
    node_names = [cna_tree.idx_to_ete[i].name for i in range(cna_tree.n_nodes)]

    sample_out = tmp_path / "S1" / "sntree" / "soft_em"
    sample_out.mkdir(parents=True)
    pkl = sample_out / "soft_em_results.pkl"
    with open(pkl, "wb") as f:
        pickle.dump({
            "pi_b":       np.full(cna_tree.n_nodes, 1.0 / cna_tree.n_nodes),
            "pi_0":       0.5,
            "node_names": node_names,
            "alpha":      ALPHA,
            "beta":       BETA,
            "history":    [],
            "p0":         P0,
            "p1_fp_mode": "one_over_c",
            "tree_path":  "stub.new",
        }, f)

    monkeypatch.setattr(
        score_mod, "load_structures",
        lambda input_paths, tree_path: (cna_tree, snv_dataset, transitions),
    )

    class _Paths:
        preprocessed_tree = "stub.new"

    config = Config()
    config.batch_size = 2
    return dict(
        output_root=str(tmp_path), config=config, input_paths=_Paths(),
        out_dir=str(sample_out),
    )


def test_run_score_reproduces_the_map_placements(scoring_run):
    res = run_score("S1", scoring_run["output_root"], scoring_run["config"],
                    scoring_run["input_paths"])

    observed = res["observed"]
    assert observed.loc["clade", "node"] == "n2_0"
    assert observed.loc["empty", "node"] == "Null"
    assert os.path.exists(os.path.join(res["output_dir"], "llr_observed.tsv"))
    # Written alongside the parameters it scored
    assert res["output_dir"] == scoring_run["out_dir"]
    assert res["summary"] is None


def test_run_score_permutation_writes_a_null_per_replicate(scoring_run):
    res = run_score("S1", scoring_run["output_root"], scoring_run["config"],
                    scoring_run["input_paths"],
                    permute=True, seed=3, replicates=2)

    assert sorted(res["permuted"]) == ["permuted_seed3", "permuted_seed4"]
    for label in res["permuted"]:
        assert os.path.exists(
            os.path.join(res["output_dir"], f"llr_{label}.tsv")
        )

    summary = res["summary"]
    assert list(summary[summary.layer == "all"]["quantile"]) == NULL_QUANTILES
    assert "subclonal" in set(summary["layer"])
    assert os.path.exists(os.path.join(res["output_dir"], "llr_summary.tsv"))
    assert os.path.exists(os.path.join(res["output_dir"], "llr_status_counts.tsv"))

    # The clade locus loses its advantage under the shuffle
    for df in res["permuted"].values():
        assert df.loc["clade", "llr_null"] < res["observed"].loc["clade", "llr_null"]


def test_run_score_subsample_scores_the_same_loci_in_every_pass(scoring_run):
    res = run_score("S1", scoring_run["output_root"], scoring_run["config"],
                    scoring_run["input_paths"],
                    permute=True, replicates=2,
                    subsample_loci=2, subsample_seed=11)

    assert len(res["observed"]) == 2
    for df in res["permuted"].values():
        assert sorted(df.index) == sorted(res["observed"].index)


def test_run_score_rejects_a_mismatched_tree(scoring_run, monkeypatch):
    """Scoring against a tree the parameters were not fit on must not proceed."""
    pkl = os.path.join(scoring_run["out_dir"], "soft_em_results.pkl")
    with open(pkl, "rb") as f:
        res = pickle.load(f)
    res["node_names"] = [f"other_{n}" for n in res["node_names"]]
    with open(pkl, "wb") as f:
        pickle.dump(res, f)

    with pytest.raises(RuntimeError, match="--refined-tree"):
        run_score("S1", scoring_run["output_root"], scoring_run["config"],
                  scoring_run["input_paths"])


def test_run_score_requires_existing_results(tmp_path):
    with pytest.raises(RuntimeError, match="soft-em"):
        run_score("S1", str(tmp_path), Config(), None)


def test_written_tsv_round_trips_through_pandas(scoring_run):
    """
    Non-finite values are written raw, so a reader must be able to recover
    them — inf as inf, NaN via na_rep, and the "Null" sentinel untouched.
    """
    res = run_score("S1", scoring_run["output_root"], scoring_run["config"],
                    scoring_run["input_paths"])
    path = os.path.join(res["output_dir"], "llr_observed.tsv")

    back = pd.read_csv(path, sep="\t", index_col="snv")

    assert list(back.columns) == ["node", "llr_null", "llr_margin", "score_status"]
    assert back["llr_null"].dtype.kind == "f"
    assert back.loc["empty", "node"] == "Null"
    np.testing.assert_allclose(
        back["llr_null"].to_numpy(), res["observed"]["llr_null"].to_numpy(),
        rtol=1e-5,
    )
