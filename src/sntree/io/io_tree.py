import dendropy
from ete4 import Tree

def read_tree(path,
              remove_diploid=True, diploid_name="diploid"):
  # Imports the medicc2 tree in Newick format, resolves multifurcations
  # and removes 0-length edges, roots to diploid, and optionally removes
  # diploid from the tree

  # import into dendropy to resolve any multifurcations and remove internal nodes that have 0 edge length
  t = dendropy.Tree.get(path=path, schema="newick")
  #t.reroot_at_node(t.find_node_with_taxon_label("diploid"), update_bipartitions=False)
  t.collapse_unweighted_edges()
  t.suppress_unifurcations()
  t.as_string(schema="newick")
  t = Tree(t.as_string(schema="newick"), parser=1)

  # Root it to the diploid
  t.set_outgroup(t[diploid_name])
  # Detach the diploid just to avoid issues
  if remove_diploid:
    t[diploid_name].detach()

  t.name = "root"
  return t

