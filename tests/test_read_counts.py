"""Read-count storage must stay compact without changing counts or wrapping."""

import numpy as np
import pandas as pd
import pytest
from cyvcf2 import VCF

from sntree.io.io_preprocess import build_snv_dataset
from sntree.io.io_snv import (
    vcf_list_to_tables,
    vcf_subset_to_tables,
    vcf_to_tables,
)


def _write_vcf(path, records):
    path.write_text(
        '##fileformat=VCFv4.2\n'
        '##contig=<ID=1,length=1000000>\n'
        '##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths">\n'
        '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tc1\tnormal\tc2\n'
        + records
    )
    return str(path)


@pytest.fixture
def vcf_path(tmp_path):
    return _write_vcf(
        tmp_path / "counts.vcf",
        "1\t100\t.\tA\tC\t.\tPASS\t.\tAD\t9,1\t20,0\t0,0\n"
        "1\t200\t.\tG\tA,T\t.\tPASS\t.\tAD\t5,2,0\t12,0,1\t2,0,3\n"
        "1\t300\t.\tA\tAC\t.\tPASS\t.\tAD\t8,2\t9,0\t0,0\n"
        "1\t400\t.\tA\tG\t.\tPASS\t.\tAD\t0,0\t0,4\t0,0\n",
    )


def test_vcf_counts_and_dataset_preserve_values_in_int16(vcf_path):
    vcf = VCF(vcf_path)
    try:
        ids, ref, alt, normal_ref, normal_alt = vcf_to_tables(
            vcf, normal_name="normal"
        )
    finally:
        vcf.close()

    assert ids == ["1:100:A>C", "1:200:G>A", "1:200:G>T"]
    for table in (ref, alt):
        assert table.to_numpy().dtype == np.int16
    assert normal_ref.dtype == normal_alt.dtype == np.int16
    assert normal_ref.tolist() == [20, 12, 12]
    assert normal_alt.tolist() == [0, 0, 1]
    np.testing.assert_array_equal(ref, [[9, 0], [5, 2], [5, 2]])
    np.testing.assert_array_equal(alt, [[1, 0], [2, 0], [0, 3]])

    metadata = pd.DataFrame({"snv": ids, "cna_idx": [0, 0, 0]})
    data = build_snv_dataset(ref, alt, metadata, ["c2", "c1"])
    assert data.ks.dtype == data.ns.dtype == np.int16
    np.testing.assert_array_equal(data.ks, [[0, 1], [0, 2], [3, 0]])
    np.testing.assert_array_equal(data.ns, [[0, 10], [2, 7], [5, 5]])
    assert data.ks.nbytes + data.ns.nbytes == 4 * 3 * 2


def test_subset_and_empty_loaders_use_int16(vcf_path):
    vcf = VCF(vcf_path)
    try:
        ids, ref, alt = vcf_subset_to_tables(
            vcf, ["1:200:G>T"], normal_name="normal",
            name_map={"c2": "mapped"}, cells_keep=["mapped"],
        )
    finally:
        vcf.close()
    assert ids == ["1:200:G>T"]
    assert list(ref.columns) == ["mapped"]
    assert ref.to_numpy().dtype == alt.to_numpy().dtype == np.int16
    assert ref.iloc[0, 0] == 2
    assert alt.iloc[0, 0] == 3

    for subset in (False, True):
        vcf = VCF(vcf_path)
        try:
            if subset:
                result = vcf_subset_to_tables(vcf, [])
            else:
                result = vcf_to_tables(vcf, min_cells=2, normal_name="normal")
        finally:
            vcf.close()
        assert result[0] == []
        for table in result[1:]:
            assert table.to_numpy().dtype == np.int16


def test_multiple_vcfs_keep_int16(vcf_path):
    vcfs = [VCF(vcf_path), VCF(vcf_path)]
    try:
        ids, *tables = vcf_list_to_tables(vcfs, normal_name="normal")
    finally:
        for vcf in vcfs:
            vcf.close()
    assert len(ids) == 6
    assert all(table.to_numpy().dtype == np.int16 for table in tables)


@pytest.mark.parametrize("fill", [0, 7])
@pytest.mark.parametrize("normal_ad,expected", [
    (".", (None, None)),
    (".,.", (None, None)),
    (".,3", (None, 3)),
    ("97,.", (97, None)),
    ("97", (97, None)),
])
def test_missing_normal_counts_use_fill_and_remain_int16(
    tmp_path, normal_ad, expected, fill,
):
    path = _write_vcf(
        tmp_path / "missing_normal.vcf",
        f"1\t100\t.\tA\tC\t.\tPASS\t.\tAD\t9,1\t{normal_ad}\t0,1\n",
    )
    vcf = VCF(path)
    try:
        ids, ref, alt, normal_ref, normal_alt = vcf_to_tables(
            vcf, normal_name="normal", fill_missing_normal_with=fill,
        )
    finally:
        vcf.close()
    assert ids == ["1:100:A>C"]
    assert normal_ref.dtype == normal_alt.dtype == np.int16
    assert (normal_ref.iloc[0], normal_alt.iloc[0]) == tuple(
        fill if value is None else value for value in expected
    )
    np.testing.assert_array_equal(ref, [[9, 0]])
    np.testing.assert_array_equal(alt, [[1, 1]])


@pytest.mark.parametrize("normal_ad", ["-1,0", "0,-1", "32768,0", "0,32768"])
def test_invalid_normal_counts_still_raise(tmp_path, normal_ad):
    path = _write_vcf(
        tmp_path / "invalid_normal.vcf",
        f"1\t100\t.\tA\tC\t.\tPASS\t.\tAD\t9,1\t{normal_ad}\t0,1\n",
    )
    vcf = VCF(path)
    try:
        with pytest.raises(ValueError, match="Normal .* counts.*32767"):
            vcf_to_tables(vcf, normal_name="normal")
    finally:
        vcf.close()


@pytest.mark.parametrize("counts,expected", [
    (".", (0, 0)), (".,.", (0, 0)), (".,4", (0, 4)), ("6,.", (6, 0)), ("6", (6, 0)),
])
@pytest.mark.parametrize("subset", [False, True])
def test_missing_cell_counts_fill_zero(tmp_path, counts, expected, subset):
    path = _write_vcf(
        tmp_path / "missing_cell.vcf",
        f"1\t100\t.\tA\tC\t.\tPASS\t.\tAD\t{counts}\t0,0\t0,1\n",
    )
    vcf = VCF(path)
    try:
        if subset:
            _, ref, alt = vcf_subset_to_tables(vcf, ["1:100:A>C"], normal_name="normal")
        else:
            _, ref, alt, _, _ = vcf_to_tables(vcf, normal_name="normal")
    finally:
        vcf.close()
    assert ref.to_numpy().dtype == alt.to_numpy().dtype == np.int16
    np.testing.assert_array_equal(ref, [[expected[0], 0]])
    np.testing.assert_array_equal(alt, [[expected[1], 1]])


@pytest.mark.parametrize("counts", ["32768,1", "-1,1"])
@pytest.mark.parametrize("subset", [False, True])
def test_vcf_rejects_overflow_and_negative_counts(tmp_path, counts, subset):
    path = _write_vcf(
        tmp_path / "invalid.vcf",
        f"1\t100\t.\tA\tC\t.\tPASS\t.\tAD\t{counts}\t0,0\t0,1\n",
    )
    vcf = VCF(path)
    try:
        with pytest.raises(ValueError, match="32767"):
            if subset:
                vcf_subset_to_tables(vcf, ["1:100:A>C"], normal_name="normal")
            else:
                vcf_to_tables(vcf, normal_name="normal")
    finally:
        vcf.close()


@pytest.mark.parametrize("dtype", [np.int16, np.int32, np.int64])
def test_total_depth_boundary_and_chunk_boundary(dtype):
    # More than one conversion batch, including the highest valid depth.
    ids = [f"snv{i}" for i in range(1025)]
    ref = pd.DataFrame(32766, index=ids, columns=["c1"], dtype=dtype)
    alt = pd.DataFrame(1, index=ids, columns=["c1"], dtype=dtype)
    metadata = pd.DataFrame({"snv": ids, "cna_idx": 0})
    data = build_snv_dataset(ref, alt, metadata, ["c1"])
    assert data.ks.dtype == data.ns.dtype == np.int16
    assert np.all(data.ns == 32767)

    # Each input fits in int16; their sum does not.
    alt.iloc[-1, 0] = 2
    with pytest.raises(ValueError, match="Total read depth.*32767"):
        build_snv_dataset(ref, alt, metadata, ["c1"])


@pytest.mark.parametrize("invalid", [32768, -1, 0.5, np.nan])
def test_dataset_rejects_invalid_counts_from_external_tables(invalid):
    ref = pd.DataFrame({"c1": [invalid]}, index=["snv"])
    alt = pd.DataFrame({"c1": [1]}, index=["snv"])
    metadata = pd.DataFrame({"snv": ["snv"], "cna_idx": [0]})
    with pytest.raises(ValueError, match="Reference counts"):
        build_snv_dataset(ref, alt, metadata, ["c1"])
