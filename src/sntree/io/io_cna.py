import pandas as pd

def import_cna_data(mapping_file,
                    cna_file):
  # Takes file paths from chisel and medicc2 outputs
  # mapping_file: ex.info.tsv from chisel
  # cna_file: ex_final_cn_profiles.tsv from medicc2

  sample_mapping = pd.read_csv(mapping_file, sep = "\t", dtype = str)
  cna_profiles = pd.read_csv(cna_file, sep = "\t")

  # Create CHISEL Barcode to Cell Name Mapping -----------------------------------------------
  sample_mapping = sample_mapping.rename(columns={sample_mapping.columns[0]: sample_mapping.columns[0].lstrip("#")})
  # Keep just what we need
  sample_mapping = sample_mapping[["CELL", "BARCODE"]]

  return sample_mapping, cna_profiles

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

def ensure_cn_tot(cna_profiles):
  cna_profiles = cna_profiles.copy()

  if 'cn_tot' in cna_profiles.columns:
    return cna_profiles

  if 'cn_total' in cna_profiles.columns:
    cna_profiles['cn_tot'] = cna_profiles['cn_total']
    return cna_profiles

  if {'cn_a', 'cn_b'}.issubset(cna_profiles.columns):
    cna_profiles['cn_tot'] = cna_profiles['cn_a'] + cna_profiles['cn_b']
    return cna_profiles

  raise ValueError(
    "CNA profiles must contain cn_tot, cn_total, or both cn_a and cn_b"
  )


def add_cna(tree, sample_mapping, cna_profiles):

  # takes mapping between cell names and cell barcodes, and CNA profile table
  # and places them on the tree
  cna_profiles = ensure_cn_tot(cna_profiles)

  # Add Root CN to the tree as Diploid (for now)  -------------------------------
  root_profile=cna_profiles.loc[cna_profiles['sample_id'] == "diploid", ['idx', 'cn_tot']].copy()
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

    cna_profile = cna_profiles.loc[cna_profiles['sample_id'] == n.name, ['idx', 'cn_tot']].copy()
    cna_dict = cna_profile.set_index('idx').to_dict(orient='index')

    n.add_prop("CN_profile", cna_dict)

  # CN parent-child difference --------------------------------------------------
  # Manually get the difference between the parent total CN and the child total CN as the edge event
  for n in tree.traverse():
    # Skip the root
    if n == tree:
      continue
    
    cna_diff_dict = {i : n.get_prop('CN_profile')[i]['cn_tot'] - n.parent.get_prop('CN_profile')[i]['cn_tot'] for i in n.get_prop('CN_profile').keys()}
    n.add_prop('CN_diff', cna_diff_dict)
    
  return tree 
  
