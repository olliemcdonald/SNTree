# sntree/io/io_tree.py

from ete4 import Tree


# ---------------------------------------------------------
# 1. Load raw MEDICC2 tree and normalize structure
# ---------------------------------------------------------

def read_raw_medicc_tree(
    path: str,
    diploid_name: str = "diploid",
    remove_diploid: bool = True,
) -> Tree:
    """
    Load MEDICC2 Newick tree and normalize:
      - Root to diploid
      - Optionally remove diploid
    Returns an ete4 Tree object.
    """

    # Load with ete4
    t = Tree(open(path).read().strip(), parser=1)

    # Root to diploid and remove if present
    if diploid_name in t:
        t.set_outgroup(t[diploid_name])

        if remove_diploid:
            t[diploid_name].detach()

    # Explicitly name root
    t.name = "root"

    return t


# ---------------------------------------------------------
# 2. Load already preprocessed tree (for ML/EM/refine)
# ---------------------------------------------------------

def read_preprocessed_tree(path: str) -> Tree:
    """
    Load a preprocessed tree written by preprocess stage.
    No structural modification is performed.
    """
    t = Tree(open(path).read().strip(), parser=1)

    # Ensure root is named
    if not t.name:
        t.name = "root"

    return t