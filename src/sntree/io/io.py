import gzip
import pickle
from ete4 import Tree 

def save_state(
    outfile,
    newick,
    sample_mapping,
    cna_profiles,
    cna_events,
    ref_df,
    alt_df,
    snv_df,
    transitions,
    placements_maxlik=None,
    predicted_scsnv=None,
):
    """
    Save core reproducible inputs + CNA-aware placements into a gzipped pickle.
    Diploid results are intentionally NOT saved here.
    """

    state = {
        "newick": newick,
        "sample_mapping": sample_mapping,
        "cna_profiles": cna_profiles,
        "cna_events": cna_events,
        "ref_df": ref_df,
        "alt_df": alt_df,
        "snv_df": snv_df,
        "transitions": transitions,
        "placements_maxlik": placements_maxlik,
        "predicted_scsnv": predicted_scsnv,
    }

    # Gzip + pickle (fast compression)
    with gzip.open(outfile, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_state(infile):
    """
    Load gzipped pickle created by save_state() and rebuild the raw tree.
    Returns:
        t (ETE Tree), sample_mapping, cna_profiles, cna_events,
        ref_df, alt_df, snv_df, transitions,
        placements_maxlik, predicted_scsnv
    """
    with gzip.open(infile, "rb") as f:
        state = pickle.load(f)

    # Reconstruct ETE tree from saved newick string
    t = Tree(state["newick"], parser=1)

    # Root it to the diploid
    t.set_outgroup(t["diploid"])
    # Detach the diploid just to avoid issues
    t["diploid"].detach()
    t.name = "root"

    return (
        t,
        state["sample_mapping"],
        state["cna_profiles"],
        state["cna_events"],
        state["ref_df"],
        state["alt_df"],
        state["snv_df"],
        state["transitions"],
        state["placements_maxlik"],
        state["predicted_scsnv"],
    )

# Simple load_state usage
"""
from sntree.io.io_state import load_state
import sntree.io.io_cna as cna
import sntree.io.io_snv as snv
from sntree.io.io_preprocess import build_all

# Load state
(
    t,                # ETE tree rebuilt from Newick
    sample_mapping,
    cna_profiles,
    cna_events,
    ref_df,
    alt_df,
    snv_df,
    transitions,
    placements_maxlik,
    predicted_scsnv,
) = load_state("/path/to/sample/state.pkl.gz")

# Rebuild annotated tree and derived structures
t = cna.add_cna(t, sample_mapping, cna_profiles, cna_events)
t = snv.add_snvs(t, ref_df, alt_df)

cna_tree, snv_dataset, transitions2 = build_all(
    ete_tree=t,
    cna_profiles=cna_profiles,
    sample_mapping=sample_mapping,
    ref_df=ref_df,
    alt_df=alt_df,
    snv_df=snv_df
)
"""