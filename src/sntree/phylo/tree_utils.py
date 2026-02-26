from typing import Dict
from ete4 import Tree # type: ignore

def _attach_constant_cn(tree: Tree, cn_profile: Dict[int, Dict[str, int]]) -> None:
    for n in tree.traverse():
        n.add_prop("CN_profile", cn_profile)


def _set_leaf_cell_ids(tree: Tree) -> None:
    for n in tree.traverse():
        if n.is_leaf:
            n.add_prop("cell_id", n.name)


def _tree_to_newick(tree: Tree) -> str:
    return tree.write(parser=1).strip()

