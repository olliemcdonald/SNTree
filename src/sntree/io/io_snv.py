import os
import re
from cyvcf2 import VCF # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore

def _map_sample_names(samples, name_map=None):
    if not name_map:
        return list(samples)
    return [name_map.get(s, s) for s in samples]

def import_snv_vcfs(folder_path, pattern="_somatic.pileup.vcf.gz"):

    # Find files containing the pattern
    vcf_files = [f for f in os.listdir(folder_path) if pattern in f]

    # Sort by chromosome number extracted after "chr"
    def extract_chr_number(filename):
        match = re.search(r'chr(\d+|X|Y|M)', filename)
        if match:
            val = match.group(1)
            # Convert numeric chromosomes to int; keep X/Y/M ordered after numbers
            if val.isdigit():
                return (0, int(val))
            return (1, {"X": 23, "Y": 24, "M": 25}.get(val, 99))
        return (2, 999)

    vcf_files = sorted(vcf_files, key=extract_chr_number)

    vcf_list = []
    for vcf_file in vcf_files:
        vcf_path = os.path.join(folder_path, vcf_file)
        vcf_list.append(VCF(vcf_path))

    return vcf_list



def vcf_to_tables(
    vcf,
    min_cells=1,
    normal_name="/michorlab/mcdonald/single_cell_cna/data/pancancer/C1/normal_cells/C1_normal_markdup.bam",
    fill_missing_normal_with=0,  # keep shapes stable without using normal yet
):
    # vcf: cyvcf2 VCF object
    # min_cells: minimum number of cells that must have >0 alt reads to keep the variant
    # normal_name: sample name of the normal cell (if any) in the VCF ex. "/michorlab/mcdonald/single_cell_cna/data/pancancer/C1/normal_cells/C1_normal_markdup.bam"
    # fill_missing_normal_with: value to fill in for normal ref/alt counts if normal is not present
    samples = list(vcf.samples)

    # Figure out normal (optional) and cells
    if normal_name is not None and normal_name in samples:
        normal_idx = samples.index(normal_name)
        cell_idx = [i for i in range(len(samples)) if i != normal_idx]
        has_normal = True
    else:
        normal_idx = None
        cell_idx = list(range(len(samples)))
        has_normal = False

    cell_names = [samples[i] for i in cell_idx]

    variant_ids, ref_rows, alt_rows, norm_ref, norm_alt = [], [], [], [], []

    for rec in vcf:
        # Only SNVs
        if not rec.REF or len(rec.REF) != 1:
            continue

        AD = rec.format("AD")
        if AD is None:
            continue
        if AD.ndim != 2 or AD.shape[0] != len(samples) or AD.shape[1] < 2:
            continue

        ref_all = AD[:, 0]
        n_alts = AD.shape[1] - 1

        for j, a in enumerate(rec.ALT or []):
            # skip symbolic or non-SNV
            if (a is None) or (isinstance(a, str) and a.startswith("<") and a.endswith(">")) or len(a) != 1:
                continue
            if j >= n_alts:
                continue

            alt_col = AD[:, 1 + j]

            # per-ALT filter: how many cells have >0 alt reads?
            cells_supporting = int(np.sum(alt_col[cell_idx] > 0))
            if cells_supporting < min_cells:
                continue

            variant_ids.append(f"{rec.CHROM}:{rec.POS}:{rec.REF}>{a}")
            ref_rows.append(ref_all[cell_idx])
            alt_rows.append(alt_col[cell_idx])

            if has_normal:
                norm_ref.append(int(ref_all[normal_idx]))
                norm_alt.append(int(alt_col[normal_idx]))
            else:
                norm_ref.append(fill_missing_normal_with)
                norm_alt.append(fill_missing_normal_with)

    # Return empty, well-formed tables if nothing passed
    if len(variant_ids) == 0:
        ref_df = pd.DataFrame(columns=cell_names, dtype=int)
        alt_df = pd.DataFrame(columns=cell_names, dtype=int)
        normal_ref_col = pd.Series([], dtype=int, name="normal_ref")
        normal_alt_col = pd.Series([], dtype=int, name="normal_alt")
        return variant_ids, ref_df, alt_df, normal_ref_col, normal_alt_col

    ref_df = pd.DataFrame(np.vstack(ref_rows), index=variant_ids, columns=cell_names).fillna(0).astype(int)
    alt_df = pd.DataFrame(np.vstack(alt_rows), index=variant_ids, columns=cell_names).fillna(0).astype(int)
    normal_ref_col = pd.Series(norm_ref, index=variant_ids, name="normal_ref").fillna(0).astype(int)
    normal_alt_col = pd.Series(norm_alt, index=variant_ids, name="normal_alt").fillna(0).astype(int)

    return variant_ids, ref_df, alt_df, normal_ref_col, normal_alt_col


def vcf_list_to_tables(
    vcf_list,
    min_cells=1,
    normal_name=None,
    fill_missing_normal_with=0,
):

    # If a single VCF or not a list, delegate to vcf_to_tables
    if not isinstance(vcf_list, list):
        return vcf_to_tables(
            vcf_list,
            min_cells=min_cells,
            normal_name=normal_name,
            fill_missing_normal_with=fill_missing_normal_with,
        )
    elif len(vcf_list) == 1:
        return vcf_to_tables(
            vcf_list[0],
            min_cells=min_cells,
            normal_name=normal_name,
            fill_missing_normal_with=fill_missing_normal_with,
        )

    # ---- Accumulators (fast list-based, no concat in loop) ----
    variant_ids_all = []
    ref_blocks = []
    alt_blocks = []
    norm_ref_blocks = []
    norm_alt_blocks = []

    # ---- Process each VCF ----
    for vcf in vcf_list:
        variant_ids, ref_df, alt_df, norm_ref, norm_alt = vcf_to_tables(
            vcf,
            min_cells=min_cells,
            normal_name=normal_name,
            fill_missing_normal_with=fill_missing_normal_with,
        )

        # Skip empties early → avoids concat warnings entirely
        if len(variant_ids) == 0:
            continue

        variant_ids_all.extend(variant_ids)
        ref_blocks.append(ref_df)
        alt_blocks.append(alt_df)
        norm_ref_blocks.append(norm_ref)
        norm_alt_blocks.append(norm_alt)

    # ---- If everything empty, return clean empties ----
    if len(variant_ids_all) == 0:
        return (
            [],
            pd.DataFrame(),
            pd.DataFrame(),
            pd.Series(name="normal_ref", dtype=int),
            pd.Series(name="normal_alt", dtype=int)
        )

    # ---- Final concatenations (fast, single time) ----
    ref_df_final = pd.concat(ref_blocks, axis=0)
    alt_df_final = pd.concat(alt_blocks, axis=0)
    normal_ref_final = pd.concat(norm_ref_blocks, axis=0)
    normal_alt_final = pd.concat(norm_alt_blocks, axis=0)

    return variant_ids_all, ref_df_final, alt_df_final, normal_ref_final, normal_alt_final


def vcf_subset_to_tables(
    vcf,
    snv_ids,
    normal_name=None,
    name_map=None,
    cells_keep=None,
):
    """
    Subset VCF to a specific SNV ID set and optional cell set.
    Returns: variant_ids, ref_df, alt_df
    """
    if not isinstance(vcf, VCF):
        vcf = VCF(vcf)

    snv_ids = set(snv_ids)
    samples = list(vcf.samples)

    if normal_name is not None and normal_name in samples:
        normal_idx = samples.index(normal_name)
    else:
        normal_idx = None

    cell_idx = [i for i in range(len(samples)) if i != normal_idx]
    samples_mapped = _map_sample_names(samples, name_map)
    cell_names = [samples_mapped[i] for i in cell_idx]

    if cells_keep is not None:
        keep_set = set(cells_keep)
        keep_mask = [n in keep_set for n in cell_names]
        cell_idx = [i for i, keep in zip(cell_idx, keep_mask) if keep]
        cell_names = [n for n in cell_names if n in keep_set]

    variant_ids, ref_rows, alt_rows = [], [], []

    for rec in vcf:
        if not rec.REF or len(rec.REF) != 1:
            continue

        AD = rec.format("AD")
        if AD is None:
            continue
        if AD.ndim != 2 or AD.shape[0] != len(samples) or AD.shape[1] < 2:
            continue

        ref_all = AD[:, 0]
        n_alts = AD.shape[1] - 1

        for j, a in enumerate(rec.ALT or []):
            if (a is None) or (isinstance(a, str) and a.startswith("<") and a.endswith(">")) or len(a) != 1:
                continue
            if j >= n_alts:
                continue

            snv_id = f"{rec.CHROM}:{rec.POS}:{rec.REF}>{a}"
            if snv_id not in snv_ids:
                continue

            alt_col = AD[:, 1 + j]
            variant_ids.append(snv_id)
            ref_rows.append(ref_all[cell_idx])
            alt_rows.append(alt_col[cell_idx])

    if len(variant_ids) == 0:
        ref_df = pd.DataFrame(columns=cell_names, dtype=int)
        alt_df = pd.DataFrame(columns=cell_names, dtype=int)
        return variant_ids, ref_df, alt_df

    ref_df = pd.DataFrame(np.vstack(ref_rows), index=variant_ids, columns=cell_names).fillna(0).astype(int)
    alt_df = pd.DataFrame(np.vstack(alt_rows), index=variant_ids, columns=cell_names).fillna(0).astype(int)

    return variant_ids, ref_df, alt_df


def snv_bin_mapping(snv_df, cna_idx, warn=True):
    snv_chr_list = []

    # Identify chromosome sets
    cna_chrs = set(cna_idx["chrom"].unique())
    snv_chrs = snv_df["chrom"].unique()

    # Warn about missing chromosomes
    missing = [c for c in snv_chrs if c not in cna_chrs]
    if warn and missing:
        print(f"[WARNING] Chromosomes in SNVs but not in CNA index: {missing}. "
              "SNVs on these chromosomes will be dropped.")

    # Process by chromosome
    for chrom, snv_sub in snv_df.groupby("chrom", sort=False):

        # Skip chromosomes not in CNA
        if chrom not in cna_chrs:
            continue

        cna_sub = cna_idx[cna_idx["chrom"] == chrom]

        merged = pd.merge_asof(
            snv_sub.sort_values("pos"),
            cna_sub.sort_values("start"),
            by="chrom",
            left_on="pos",
            right_on="start",
            direction="backward"
        )

        # Mask SNVs that fall outside bins (pos >= end)
        merged.loc[merged["pos"] >= merged["end"], "idx"] = pd.NA

        snv_chr_list.append(
            merged.drop(columns=["start", "end"])
        )

    # If we dropped everything
    if not snv_chr_list:
        if warn:
            print("[WARNING] All SNVs dropped because no chromosomes matched CNA index.")
        return snv_df.assign(cna_idx=pd.NA)

    snv_df_out = pd.concat(snv_chr_list, ignore_index=True)
    snv_df_out = snv_df_out.rename(columns={"idx": "cna_idx"})

    # Count and drop rows with invalid/missing mapping
    invalid_mask = snv_df_out["cna_idx"].isna()
    n_invalid = invalid_mask.sum()

    if n_invalid > 0 and warn:
        print(f"[WARNING] {n_invalid} SNVs did not map to any CNA bin (likely outside bin ranges) "
              "and were removed.")

    snv_df_out = snv_df_out.loc[~invalid_mask].copy()

    # Now safe to convert to integer
    snv_df_out["cna_idx"] = snv_df_out["cna_idx"].astype(int)

    return snv_df_out


def snv_lookups(variant_ids, cna_idx, ref_df=None, alt_df=None,
                normal_ref=None, normal_alt=None, warn=True):

    # Build SNV table from IDs
    snv_df = pd.DataFrame({"snv": variant_ids})
    snv_df = snv_df.join(
        snv_df["snv"].str.split('[:,>]', expand=True)
        .set_axis(["chrom","pos","ref","alt"], axis=1)
    )
    snv_df["pos"] = snv_df["pos"].astype(int)
    snv_df = snv_df.sort_values(["chrom","pos"])

    # Map SNVs to CNA bins (removes invalid SNVs)
    snv_df = snv_bin_mapping(snv_df, cna_idx, warn=warn)

    # --- NEW: propagate filtering to all related tables ---
    filtered_snvs = snv_df["snv"].tolist()

    if ref_df is not None:
        ref_df = ref_df.loc[filtered_snvs]

    if alt_df is not None:
        alt_df = alt_df.loc[filtered_snvs]

    if normal_ref is not None:
        normal_ref = normal_ref.loc[filtered_snvs]

    if normal_alt is not None:
        normal_alt = normal_alt.loc[filtered_snvs]

    # Build SNV lookup dictionary
    snv_dict = (
        snv_df
        .set_index("snv")[["chrom","pos","ref","alt","cna_idx"]]
        .to_dict(orient="index")
    )

    return snv_df, snv_dict, ref_df, alt_df, normal_ref, normal_alt


def _build_dataset_for_group(
    ref_df: pd.DataFrame,
    alt_df: pd.DataFrame,
    cna_idx: pd.DataFrame,
):
    variant_ids = list(ref_df.index)
    snv_df, _, ref_df, alt_df, _, _ = snv_lookups(variant_ids, cna_idx, ref_df=ref_df, alt_df=alt_df)
    return snv_df, ref_df, alt_df


def add_snvs(tree, ref_df, alt_df):
  # Store a dict of [SNV ID: (k_alt, k_tot)]
  for n in tree.leaves():
    if n.name == 'diploid':
      continue
    n.add_prop("ref", ref_df[n.props['cell_id']].to_dict())
    n.add_prop("alt", alt_df[n.props['cell_id']].to_dict())

  return tree

 
