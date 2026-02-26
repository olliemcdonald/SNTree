import pandas as pd

def import_cna_data(mapping_file,
                    cna_file,
                    cna_events_file=None):
  # Takes file paths from chisel and medicc2 outputs
  # mapping_file: ex.info.tsv from chisel
  # cna_file: ex_final_cn_profiles.tsv from medicc2
  # cna_events_file: ex_copynumber_events_df.tsv from medicc2

  sample_mapping = pd.read_csv(mapping_file, sep = "\t", dtype = str)
  cna_profiles = pd.read_csv(cna_file, sep = "\t")
  if cna_events_file is not None:
    cna_events = pd.read_csv(cna_events_file, sep="\t")

  # Create CHISEL Barcode to Cell Name Mapping -----------------------------------------------
  sample_mapping = sample_mapping.rename(columns={sample_mapping.columns[0]: sample_mapping.columns[0].lstrip("#")})
  # Keep just what we need
  sample_mapping = sample_mapping[["CELL", "BARCODE"]]

  return sample_mapping, cna_profiles, cna_events

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


def add_cna(tree, sample_mapping, cna_profiles):
  
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
  
