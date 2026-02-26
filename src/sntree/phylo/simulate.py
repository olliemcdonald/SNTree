import numpy as np
import pandas as pd
import math
import random
from ete4 import Tree
#from ete4.treeview import TreeStyle, NodeStyle, TextFace


def generate_tree(cell_count, rng=None):
  if rng is None:
    rng = random.Random()
    # reseed random so ETE4 behaves deterministically
  random.setstate(rng.getstate())

  tree = Tree()
  tree.populate(cell_count)
  i=0
  for node in tree.traverse():
    if not node.is_leaf:
      node.name=f"v{i}"
      i+=1
  
  rng.setstate(random.getstate()) # Advance RNG
  
  return tree


def simulate_cna_profiles(
    tree,
    n_segments,
    p_gain=0.1,
    p_loss=0.1,
    wgd_prob=0.0,
    wgd_once=True,
    allow_multi_step=False,
    gain_max_step=1,
    loss_max_step=1,
    rng=None
):
    """
    Simulate CN evolution along a tree with:
      - Optional WGD events (once or multiple times)
      - Gains/losses independent of WGD
      - CN starts at diploid at the root
      - Gains and losses occur one segment at a time

    NEW BEHAVIOR:
        WGD is *independent* of gains/losses:
           First apply WGD (if it happens), THEN apply gain/loss/none.
    """

    if rng is None:
        rng = random.Random()

    # ---------- Initialize root CN ----------
    root = tree
    root.props["CN_profile"] = {
        seg: {"cn_tot": 2, "cn_a": 1, "cn_b": 1}
        for seg in range(n_segments)
    }

    wgd_used = False
    events = []

    # ---------- Traverse tree ----------
    for node in tree.traverse("preorder"):
        if node.is_root:
            continue

        parent = node.up
        parent_CN = parent.props["CN_profile"]

        # Copy parent CN
        node.props["CN_profile"] = {
            seg: dict(parent_CN[seg])  # full CN dict
            for seg in range(n_segments)
        }

        # ---------------------------------------------------------
        # STEP A: WGD is independent
        # ---------------------------------------------------------
        do_wgd = False
        if wgd_prob > 0:
            if not wgd_once or not wgd_used:
                if rng.random() < wgd_prob:
                    do_wgd = True
                    wgd_used = True

        if do_wgd:
            for seg in range(n_segments):
                cn_before = node.props["CN_profile"][seg]["cn_tot"]
                cn_after = cn_before * 2
                node.props["CN_profile"][seg]["cn_tot"] = cn_after
                events.append((
                    node.name, "WGD", seg, 2, cn_before, cn_after
                ))

        # ---------------------------------------------------------
        # STEP B: Gains/losses occur *after* WGD, independently
        # ---------------------------------------------------------
        r = rng.random()
        p_none = 1 - p_gain - p_loss

        if r < p_gain:
            event_type = "GAIN"
        elif r < p_gain + p_loss:
            event_type = "LOSS"
        else:
            event_type = "NONE"

        if event_type in ("GAIN", "LOSS"):
            seg = rng.integers(0, n_segments - 1)
            cn_before = node.props["CN_profile"][seg]["cn_tot"]

            if allow_multi_step:
                if event_type == "GAIN":
                    step = rng.integers(1, gain_max_step)
                else:
                    step = -rng.integers(1, loss_max_step)
            else:
                step = +1 if event_type == "GAIN" else -1

            cn_after = max(0, cn_before + step)
            node.props["CN_profile"][seg]["cn_tot"] = cn_after

            events.append((
                node.name,
                event_type,
                seg,
                step,
                cn_before,
                cn_after
            ))

    # ---------- Build CNA DataFrame ----------
    rows = []
    for node in tree.traverse():
        for seg in range(n_segments):
            d = node.props["CN_profile"][seg]
            rows.append({
                "sample_id": node.name,
                "idx": seg,
                "cn_tot": d["cn_tot"],
                "cn_a": d["cn_a"],
                "cn_b": d["cn_b"],
            })

    cna_df = pd.DataFrame(rows)
    return events, cna_df


def simulate_cna_profiles_gainsonly(tree, n_segments, p_gain=0.1, delta=1, rng=None):
    """
    Simulate CN profiles for every node and segment.
    Returns:
        cna_profiles_sim : DataFrame with columns:
            sample_id, seg, cn_a, cn_b, cn_tot
    """

    if rng is None:
        rng = random.Random()

    # initialize diploid
    for node in tree.traverse():
        node.props["CN_profile"] = {seg: {"cn_tot": 2, "cn_a": 1, "cn_b": 1}
                                    for seg in range(n_segments)}

    # simulate gains recursively
    events = []
    for node in tree.traverse("preorder"):
        for seg in range(n_segments):

            if rng.random() < p_gain:
                # apply gain to node and all descendants
                for n2 in node.traverse():
                    n2.props["CN_profile"][seg]["cn_tot"] += delta

                events.append((node.name, seg))

    # build DataFrame
    rows = []
    for node in tree.traverse():
        for seg in range(n_segments):
            d = node.props["CN_profile"][seg]
            rows.append({
                "sample_id": node.name,
                "idx": seg,
                "cn_tot": d["cn_tot"],
                "cn_a": d["cn_a"],
                "cn_b": d["cn_b"],
            })

    return events, pd.DataFrame(rows)


def simulate_snv_evolution(tree, snv_ids, snv_cna_idx, p_origin=0.05, rng=None):
    """
    Simulate SNV multiplicities along a tree that already has CN profiles,
    including:
        - CN gains (delta > 0)
        - CN losses (delta < 0)
        - multi-step changes (|delta| > 1)
        - WGD (handled via CN doubling before simulation)
    Returns:
        origins: list of (snv_id, origin_node_name, seg)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Initialize multiplicities
    for node in tree.traverse():
        node.props["SNV_mult"] = {snv: 0 for snv in snv_ids}

    origins = []

    # SNV-by-SNV simulation
    for snv_id, seg in zip(snv_ids, snv_cna_idx):

        # ---------------------------
        # 1. Choose origin ONCE
        # ---------------------------
        origin_node = None
        for node in tree.traverse("preorder"):
            if rng.random() < p_origin:
                origin_node = node
                origins.append((snv_id, node.name, seg))
                break

        if origin_node is None:
            # SNV does not appear in tree
            continue

        # SNV originates with multiplicity = 1
        origin_node.props["SNV_mult"][snv_id] = 1

        # ---------------------------
        # 2. Propagate downwards
        # ---------------------------
        stack = [origin_node]

        while stack:
            parent = stack.pop()

            m_parent = parent.props["SNV_mult"][snv_id]
            c_parent = parent.props["CN_profile"][seg]["cn_tot"]

            for child in parent.children:

                m_child = m_parent
                c_child = child.props["CN_profile"][seg]["cn_tot"]

                delta = c_child - c_parent

                # ---------------------------
                # Handle CN gains: delta > 0
                # ---------------------------
                if delta > 0 and m_child > 0:
                    # Each new copy is mutated with prob = m_parent/c_parent
                    p_dup = m_parent / c_parent if c_parent > 0 else 0.0
                    gain_mutations = rng.binomial(delta, p_dup)
                    m_child += gain_mutations

                # ----------------------------
                # Handle CN losses: delta < 0
                # ----------------------------
                elif delta < 0 and m_child > 0:

                    # stepwise simulate each lost copy
                    lost = -delta

                    m = m_child
                    c = c_parent

                    for _ in range(lost):
                        if c <= 0:
                            m = 0
                            break

                        # with prob m/c, we lose a mutant copy
                        if rng.random() < (m / c):
                            m -= 1
                        c -= 1

                    m_child = max(m, 0)

                # ----------------------------
                # Handle CN=0 case (after WGD & losses)
                # ----------------------------
                if c_child == 0:
                    m_child = 0

                # Store updated multiplicity
                child.props["SNV_mult"][snv_id] = m_child

                # Continue propagating
                if len(child.children) > 0:
                    stack.append(child)

    return origins


def simulate_reads(tree, snv_ids, snv_cna_idx, nbinom_mu=10, nbinom_alpha=0.05, rng=None):
    """
    Produce ref_df and alt_df indexed by SNV id, columns = leaf names.
    Assumes SNV multiplicity and CN have been simulated.
    """
    if rng is None:
        rng = np.random.default_rng()

    leaves = [n for n in tree if n.is_leaf]
    leaf_names = [leaf.name for leaf in leaves]

    r = 1 / nbinom_alpha
    p = r / (r + nbinom_mu)

    alt = {leaf.name: {} for leaf in leaves}
    ref = {leaf.name: {} for leaf in leaves}

    for leaf in leaves:
        for i, snv_id in enumerate(snv_ids):
            seg = snv_cna_idx[i]
            m = leaf.props["SNV_mult"][snv_id]
            cn = leaf.props["CN_profile"][seg]["cn_tot"]
            p_alt = m / cn if cn > 0 else 0.0

            cov = rng.negative_binomial(r, p)
#            if(p_alt < 0 or p_alt > 1 or math.isnan(p_alt)):
#                print("P Error ", p_alt, " M ", m, " CN ", cn)
            alt_reads = rng.binomial(cov, p_alt)
            ref_reads = cov - alt_reads

            alt[leaf.name][snv_id] = alt_reads
            ref[leaf.name][snv_id] = ref_reads

    alt_df = pd.DataFrame(alt).T[snv_ids].T
    ref_df = pd.DataFrame(ref).T[snv_ids].T

    return ref_df, alt_df



def simulate_dataset(num_cells=100,
                     n_segments=3,
                     num_snvs=200,
                     p_gain=0.1,          # Probability of gain
                     p_loss=0.1,          # Probability of loss (gain XOR loss)
                     wgd_prob=0.0,        # Probability of WGD (can occur alongside addtl gains/losses)
                     wgd_once=True,       # Flag to keep 1 WGD in history (don't mess with for now)
                     allow_multi_step=False, # allow multiple gains/losses per step (don't mess with this for now - +/- 1 delta)
                     gain_max_step=1,        # if above the max number of gains per step (don't mess)
                     loss_max_step=1,         # same for losses
                     p_origin=0.05,       # Probability that a node is the origin for a specific SNV
                     nbinom_mu=10,        # mean read count per locus (neg binom mean)
                     nbinom_alpha=0.05,   # dispersion parameter for read counts
                     rng=None,
                     rng_tree=None):

    if rng is None:
        rng = np.random.default_rng()
    # --- generate tree ---
    tree = generate_tree(num_cells, rng=rng_tree)
    for n in tree.leaves():
        n.props['cell_id'] = n.name
  
    # --- simulate CNA ---
    cna_events, cna_profiles_sim = simulate_cna_profiles(
        tree, n_segments, p_gain=p_gain, p_loss=p_loss,
        wgd_prob=wgd_prob, wgd_once=wgd_once, allow_multi_step=allow_multi_step,
        gain_max_step=gain_max_step, loss_max_step=loss_max_step,
        rng=rng
    )

    # --- simulate SNV metadata ---
    snv_ids = [f"snv_{i}" for i in range(num_snvs)]
    snv_cna_idx = rng.integers(0, n_segments, size=num_snvs)

    # --- simulate SNV evolution ---
    snv_origins = simulate_snv_evolution(
        tree, snv_ids, snv_cna_idx, p_origin=p_origin, rng=rng
    )

    # --- simulate read counts ---
    ref_df, alt_df = simulate_reads(tree,
                                    snv_ids,
                                    snv_cna_idx,
                                    nbinom_mu=nbinom_mu,
                                    nbinom_alpha=nbinom_alpha,
                                    rng=rng)

    # --- construct SNV metadata df ---
    snv_df = pd.DataFrame({
        "snv": snv_ids,
        "cna_idx": snv_cna_idx,
        "true_origin": [None] * num_snvs
    })
    for snv_id, origin_node, seg in snv_origins:
        snv_df.loc[snv_df.snv == snv_id, "true_origin"] = origin_node

    return tree, cna_profiles_sim, snv_df, ref_df, alt_df


