from ete4.treeview import TreeStyle, NodeStyle, TextFace

def visualize_snv_origin(
    tree,
    snv_id,
    inferred_node_name=None,
    true_node_name=None,
    diploid_inferred_node_name=None,
    title=None
):
    """
    Visualize an SNV on an ETE tree:
      - Leaves carrying the SNV: red
      - Inferred MRCA node: yellow
      - True MRCA node: blue

    Args:
        tree : ete4.Tree (with props['alt'], props['ref'], SNV_mult, etc.)
        snv_id : str, SNV name
        inferred_node_name : str or None
        true_node_name : str or None
        diploid_inferred_node_name : str or None
        title : optional, title for TreeStyle
    """

    ts = TreeStyle()
    ts.show_leaf_name = True
    ts.scale = 120
    if title is not None:
        ts.title.add_face(TextFace(title, fsize=16), column=0)

    # Define colors
    COLOR_SNVPRESENT = "#ff0000"   # red
    COLOR_INFERRED   = "#ffff00"   # yellow
    COLOR_TRUE       = "#0099ff"   # blue
    COLOR_DINFERRED  = "#00ff00"   # green
    COLOR_DEFAULT    = "#000000"   # black

    # Build lookup for SNV presence in leaves
    # Use alt/ref if available; else use SNV_mult
    has_snv = {}

    for leaf in tree:
        if leaf.is_leaf:
            #if "SNV_mult" in leaf.props and snv_id in leaf.props["SNV_mult"]:
            #    # Best source: SNV multiplicity from simulation
            #    present = (leaf.props["SNV_mult"][snv_id] > 0)
            if "alt" in leaf.props:
                # Fallback: alt/ref reads
                alt = leaf.props["alt"].get(snv_id, 0)
                ref = leaf.props["ref"].get(snv_id, 0)
                present = (alt + ref > 0 and alt > 0)
            else:
                present = False

            has_snv[leaf.name] = present

    # Set node styles
    for node in tree.traverse():

        nstyle = NodeStyle()
        nstyle["size"] = 12
        nstyle["fgcolor"] = COLOR_DEFAULT

        # Leaves that have the SNV → red
        if node.is_leaf and has_snv.get(node.name, False):
            nstyle["fgcolor"] = COLOR_SNVPRESENT

        # Inferred MRCA → yellow
        if inferred_node_name is not None and node.name == inferred_node_name:
            nstyle["fgcolor"] = COLOR_INFERRED
            nstyle["size"] = 20
            node.add_face(TextFace(f"  inferred ({snv_id})", fsize=10),
                          column=0, position="branch-right")
            
        if diploid_inferred_node_name is not None and node.name == diploid_inferred_node_name:
            nstyle["fgcolor"] = COLOR_DINFERRED
            nstyle["size"] = 20
            node.add_face(TextFace(f"  dip inferred ({snv_id})", fsize=10),
                          column=0, position="branch-right")

        # True MRCA → blue
        if true_node_name is not None and node.name == true_node_name:
            nstyle["fgcolor"] = COLOR_TRUE
            nstyle["size"] = 20
            node.add_face(TextFace(f"  true ({snv_id})", fsize=10),
                          column=0, position="branch-right")

        node.set_style(nstyle)

    return ts