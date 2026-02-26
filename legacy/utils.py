# To remove after I figure out where I put everything in the main ipynb script (they're under the individual io/ files)
"""
from cyvcf2 import VCF
import numpy as np
import pandas as pd


def barcode_cell_maps(sample_mapping):
  # Dicts to map barcode ids to cell names and vice versa
  # sample_mapping comes from import_cna_data
  barcode_to_cell = dict(zip(sample_mapping["BARCODE"], sample_mapping["CELL"]))
  cell_to_barcode = dict(zip(sample_mapping["CELL"], sample_mapping["BARCODE"]))

  return barcode_to_cell, cell_to_barcode


def cna_lookups(cna_profiles):
  # CNA Bin Lookup Table ---------------------------------------------------------------
  # Create a Bin ID Lookup table for unique bins (for quick lookup later)
  chrom_list=['chr'+str(i) for i in range(1,23)]

  cna_idx = cna_profiles[['chrom', 'start', 'end']]
  cna_idx = cna_idx.drop_duplicates()
  cna_idx['chrom']=cna_idx['chrom'].astype('category').cat.reorder_categories(chrom_list[0:23])
  cna_idx=cna_idx.sort_values(["chrom", "start"])
  cna_idx['chrom']=cna_idx['chrom'].astype(str)
  # Create a dict for lookup
  cna_idx_dict = cna_idx.to_dict(orient='index')

  # DataFrame for easy reference for me
  cna_idx.insert(loc=0, column='idx', value=cna_idx.reset_index().index)

  return cna_idx, cna_idx_dict


def add_cna_bins(cna_profiles, cna_idx):
  # Merge the index set with the original to use CNA bin IDs instead of locations
  # in the tree
  cna_profiles=cna_profiles.merge(cna_idx, how='left')
  return cna_profiles


def add_cna(tree, sample_mapping, cna_profiles, cna_events):
  
  # takes mapping between cell names and cell barcodes, CNA text file, and events file
  # and places them on the tree

  # Add Root CN to the tree as Diploid (for now)  -------------------------------
  root_profile=cna_profiles.loc[cna_profiles['sample_id'] == "diploid", ['idx', 'cn_a', 'cn_b']].copy()
  root_profile.insert(loc=1, column='cn_tot', value=root_profile.cn_a + root_profile.cn_b)
  root_dict = root_profile.set_index('idx').to_dict(orient='index')
  tree.add_prop("CN_profile", root_dict) 

  # Add cell CN to tree. First adding cell name for mapping
  barcode_to_cell, _ = barcode_cell_maps(sample_mapping)
  # Give each node its copy number profile
  for n in tree.traverse():
    if n.is_root:
      n.add_prop("cell_id", None)
      continue

    n.add_prop("cell_id", barcode_to_cell.get(n.name))

    cna_profile = cna_profiles.loc[cna_profiles['sample_id'] == n.name, ['idx', 'cn_a', 'cn_b']].copy()
    cna_profile.insert(1, 'cn_tot', value=cna_profile['cn_a'] + cna_profile['cn_b'])
    cna_dict = cna_profile.set_index('idx').to_dict(orient='index')

    n.add_prop("CN_profile", cna_dict)

    
    #n.add_prop("CN_profile", cna_profiles[cna_profiles['sample_id'] == n.name])
    #n.add_prop("CN_events", cna_events[cna_events['sample_id'] == n.name])

  # CN parent-child difference --------------------------------------------------
  # Manually get the difference between the parent total CN and the child total CN as the edge event
  for n in tree.traverse():
    # Skip the root
    if n == tree:
      continue
    
    cna_diff_dict = {i : n.get_prop('CN_profile')[i]['cn_tot'] - n.parent.get_prop('CN_profile')[i]['cn_tot'] for i in n.get_prop('CN_profile').keys()}
    n.add_prop('CN_diff', cna_diff_dict)
    
  return tree 
  

def vcf_to_tables(
    vcf,
    min_cells=1,
    normal_name="XXX/example_normal_markdup.bam",
    fill_missing_normal_with=0,  # keep shapes stable without using normal yet
):
    # vcf: cyvcf2 VCF object
    # min_cells: minimum number of cells that must have >0 alt reads to keep the variant
    # normal_name: sample name of the normal cell (if any) in the VCF ex. "XXX/X_normal_markdup.bam"
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


def snv_bin_mapping(snv_df, cna_idx):
  # Map SNVs to CNA bin index
  snv_chr_list=[]
  for chrom, snv_sub in snv_df.groupby("chrom", sort=False):
    cna_sub = cna_idx[cna_idx["chrom"] == chrom]
    merged = pd.merge_asof(snv_sub, cna_sub, by = "chrom", left_on = "pos", right_on = "start", direction="backward")
    # Mask SNVs whose pos >= end
    merged.loc[merged["pos"] >= merged["end"], "idx"] = pd.NA
    snv_chr_list.append(merged.drop(columns=['start', 'end']))

  snv_df=pd.concat(snv_chr_list, ignore_index=True)
  snv_df.rename(columns = {'idx':'cna_idx'}, inplace=True)
  snv_df['cna_idx']=snv_df['cna_idx'].astype(int)

  return snv_df


def snv_lookups(variant_ids, cna_idx):
  # Create SNV position DataFrame
  snv_df = pd.DataFrame({"snv":variant_ids})
  snv_df=snv_df.join(snv_df['snv'].str.split('[:,>]', expand=True, regex=True).set_axis(["chrom", "pos", "ref", "alt"], axis=1))
  snv_df["pos"] = snv_df["pos"].astype(int)
  snv_df=snv_df.sort_values(["chrom", "pos"])

  snv_df=snv_bin_mapping(snv_df, cna_idx)
  # snv_df for going back to look, snv_dict for mapping
  snv_dict = (
    snv_df
    .set_index("snv")[["chrom","pos","ref","alt","cna_idx"]]
    .to_dict(orient="index")
  )
  return snv_df, snv_dict


def add_snvs(tree, ref_df, alt_df):
  # Store a dict of [SNV ID: (k_alt, k_tot)]
  for n in tree.leaves():
    if n.name == 'diploid':
      continue
    n.add_prop("ref", ref_df[n.props['cell_id']].to_dict())
    n.add_prop("alt", alt_df[n.props['cell_id']].to_dict())

  return tree

"""