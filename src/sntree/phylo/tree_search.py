from typing import Sequence

import numpy as np # type: ignore
from ete4 import Tree # type: ignore


def _assign_internal_names(tree: Tree, root_name: str) -> None:
    tree.root.name = root_name  # explicitly set it
    i = 0
    for n in tree.traverse():
        if n.is_leaf or n.is_root:
            continue
        if not n.name:
            n.name = f"{root_name}_n{i}"
            i += 1

def _presence_distance(pres: np.ndarray) -> np.ndarray:
    """
    pres: (N_snvs, N_cells) boolean matrix
    returns: (N_cells, N_cells) distance matrix
    """
    n_cells = pres.shape[1]
    dist = np.zeros((n_cells, n_cells), dtype=float)
    for i in range(n_cells):
        pi = pres[:, i]
        for j in range(i + 1, n_cells):
            pj = pres[:, j]
            union = np.logical_or(pi, pj)
            inter = np.logical_and(pi, pj)
            u = int(np.sum(union))
            if u == 0:
                d = 0.0
            else:
                d = 1.0 - (int(np.sum(inter)) / u)
            dist[i, j] = d
            dist[j, i] = d
    return dist


def neighbor_joining(dist: np.ndarray, labels: Sequence[str]) -> Tree:
    """
    Minimal NJ implementation for topology initialization (branch lengths omitted).
    """
    labels = list(labels)
    n = len(labels)
    if n == 1:
        return Tree(f"{labels[0]};")
    if n == 2:
        return Tree(f"({labels[0]},{labels[1]});")

    D = dist.astype(float).copy()
    clusters = labels[:]

    while len(clusters) > 2:
        m = len(clusters)
        r = np.sum(D, axis=1)
        Q = np.full((m, m), np.inf, dtype=float)
        for i in range(m):
            for j in range(i + 1, m):
                Q[i, j] = (m - 2) * D[i, j] - r[i] - r[j]

        i, j = divmod(np.argmin(Q), m)
        if i > j:
            i, j = j, i

        new_label = f"({clusters[i]},{clusters[j]})"
        new_row = []
        for k in range(m):
            if k == i or k == j:
                continue
            d = 0.5 * (D[i, k] + D[j, k] - D[i, j])
            new_row.append(d)

        # remove j and i from D and clusters
        mask = [k for k in range(m) if k not in (i, j)]
        D = D[np.ix_(mask, mask)]
        clusters = [clusters[k] for k in mask]

        # append new node
        clusters.append(new_label)
        if len(D) == 0:
            D = np.zeros((1, 1), dtype=float)
        else:
            new_row = np.array(new_row, dtype=float)
            D = np.pad(D, ((0, 1), (0, 1)), constant_values=0.0)
            D[-1, :-1] = new_row
            D[:-1, -1] = new_row

    final = f"({clusters[0]},{clusters[1]});"
    return Tree(final)



def NNI_step(tree, best_ll, score_func):
    """
    Perform one full NNI neighborhood search.
    Returns:
        best_tree (copy),
        best_ll,
        best_placements,
        improved (bool)
    """

    improved = False
    best_tree = tree
    best_placements = None

    for n in tree.descendants():
        if n.is_leaf:
            continue

        a, b = n.get_children()

        if n.up.is_root:
            if n is tree.get_children()[1]:
                continue
            c = n.get_sisters()[0].get_children()[0]
        else:
            c = n.get_sisters()[0]

        # --- First swap ---
        au = a.up
        cu = c.up

        a.detach()
        c.detach()
        au.add_child(c)
        cu.add_child(a)

        pl, ll = score_func(tree)

        if ll > best_ll:
            print(f"    -> Improvement found! {best_ll:.4f} → {ll:.4f}")
            best_ll = ll
            best_tree = tree.copy()
            best_placements = pl
            improved = True

        # restore
        a.detach()
        c.detach()
        cu.add_child(c)
        au.add_child(a)

        # --- Second swap ---
        bu = b.up

        b.detach()
        c.detach()
        bu.add_child(c)
        cu.add_child(b)

        pl, ll = score_func(tree)

        if ll > best_ll:
            print(f"    -> Improvement found! {best_ll:.4f} → {ll:.4f}")
            best_ll = ll
            best_tree = tree.copy()
            best_placements = pl
            improved = True

        # restore
        b.detach()
        c.detach()
        cu.add_child(c)
        bu.add_child(b)

    return best_tree, best_ll, best_placements, improved

