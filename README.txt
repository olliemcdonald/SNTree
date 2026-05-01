------------------------------------------------------------
sntree: Single-Cell SNV Phylogenetic Inference and Refinement
------------------------------------------------------------

sntree performs:

1) CNA-aware maximum likelihood SNV placement
2) EM estimation of sequencing error parameters (alpha, beta)
3) Local SNV-based refinement of CNA-identical subtrees

The workflow is modular. You can run each stage independently or execute the full pipeline.


------------------------------------------------------------
Installation
------------------------------------------------------------

From this directory (`src/sntree`), install into your active environment:

    python -m pip install -e .

Then run:
    sntree <command>


------------------------------------------------------------
Available Commands
------------------------------------------------------------

1) Maximum Likelihood Placement

Runs CNA-aware SNV placement using alpha_init and beta_init values
(defaults from Config unless overridden).

    sntree ml <sample> <input_root> <output_root>

Outputs are written to:

    output_root/<sample>/ml/
        ml_results.pkl


2) EM Parameter Estimation

Runs EM to estimate alpha and beta from the CNA-aware model.

    sntree em <sample> <input_root> <output_root>

Outputs are written to:

    output_root/<sample>/em/
        em_results.pkl

The EM output contains:
    - alpha
    - beta
    - placements_em


3) Subtree Refinement

Refines CNA-identical clades using SNV placements from either
the EM stage (preferred) or ML stage.

    sntree refine <sample> <input_root> <output_root>

Behavior:
    - If EM results exist, refinement uses EM placements.
    - Otherwise, if ML results exist, refinement uses ML placements.
    - If neither exists, an error is raised.

Outputs are written to:

    output_root/<sample>/refine/
        refined_full_tree.new
        group_XXX/
            refined_subtree.newick
            snv_assignments.tsv
            likelihood.txt


4) Full Pipeline

Runs EM -> Subtree refinement sequentially.

    sntree pipeline <sample> <input_root> <output_root>

Note:
    The pipeline runs EM (not ML) before refinement.
    If you want ML-based refinement, run ml first and then refine.


------------------------------------------------------------
Optional Parameter Overrides
------------------------------------------------------------

You may override default configuration values:

    sntree pipeline C2 /input /output \
        --alpha-init 0.001 \
        --beta-init 0.001 \
        --p0 0.001 \
        --batch-size 512 \
        --max-iters 100 \
        --em-max-iters 50

These override the defaults in Config.


------------------------------------------------------------
Expected Input Directory Structure
------------------------------------------------------------

<input_root>/<sample>/
    medicc2/
        <sample>_final_tree.new
        <sample>_final_cn_profiles.tsv
        <sample>_pairwise_distances.tsv
    chisel/
        <sample>.info.tsv
    snv/
        consensus_singlecell_counts.vcf.gz
    normal_cells/
        <sample>_normal_markdup.bam


------------------------------------------------------------
Output Directory Structure
------------------------------------------------------------

<output_root>/<sample>/
    ml/
    em/
    refine/

Each stage writes only its own outputs and can be rerun independently.


------------------------------------------------------------
Typical Usage
------------------------------------------------------------

Full workflow (recommended):

    sntree pipeline C2 /path/to/input /path/to/output

Manual staged workflow:

    sntree ml C2 /input /output
    sntree em C2 /input /output
    sntree refine C2 /input /output


------------------------------------------------------------
Notes
------------------------------------------------------------

- The refinement stage assumes CNA-identical groups have no CN transitions below their MRCA.
- The EM stage uses the CNA-aware likelihood model.
- The refinement stage uses a constant-CN cached likelihood for efficiency.
- ML and EM are alternative placement engines; refinement consumes their output.
- All stages are independent and resumable.

End of README.
