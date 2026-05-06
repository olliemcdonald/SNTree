import pandas as pd
import pytest

from sntree.io.io_cna import ensure_cn_tot


def test_ensure_cn_tot_from_cn_total():
    df = pd.DataFrame({"cn_total": [2, 3]})

    out = ensure_cn_tot(df)

    assert out["cn_tot"].tolist() == [2, 3]
    assert "cn_tot" not in df.columns


def test_ensure_cn_tot_from_allele_columns():
    df = pd.DataFrame({"cn_a": [1, 2], "cn_b": [1, 3]})

    out = ensure_cn_tot(df)

    assert out["cn_tot"].tolist() == [2, 5]


def test_ensure_cn_tot_keeps_existing_cn_tot():
    df = pd.DataFrame({"cn_tot": [4], "cn_total": [99]})

    out = ensure_cn_tot(df)

    assert out["cn_tot"].tolist() == [4]


def test_ensure_cn_tot_requires_total_or_alleles():
    df = pd.DataFrame({"sample_id": ["cell1"]})

    with pytest.raises(ValueError, match="cn_tot"):
        ensure_cn_tot(df)
