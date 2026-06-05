------------------------------------------------------------
sntree: Single-Cell SNV Phylogenetic Inference and Refinement
------------------------------------------------------------

sntree performs:

1) CNA-aware maximum likelihood SNV placement
2) EM estimation of sequencing error parameters (alpha, beta)
3) Soft branch-proportion inference (relative mutation burden per branch)
4) Local SNV-based refinement of CNA-identical subtrees

The workflow is modular. You can run each stage independently or execute
the full pipeline. The soft branch-proportion EM runs automatically as
part of the EM stage and produces relative branch lengths without making
hard variant calls.


------------------------------------------------------------
Installation
------------------------------------------------------------

From the project root, install into your active environment:

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

    output_root/<sample>/sntree/ml/
        placements_ml.tsv
        placements_ml_loglik.tsv
        ml_results.pkl


2) EM Parameter Estimation + Soft Branch-Proportion Inference

Runs EM to estimate alpha and beta, then automatically runs the soft
branch-proportion EM to infer relative mutation burdens pi_b per branch.

    sntree em <sample> <input_root> <output_root>

Outputs are written to:

    output_root/<sample>/sntree/em/
        placements_em.tsv           Hard MAP SNV placements
        placements_em_loglik.tsv    Per-SNV log-likelihoods
        em_results.pkl              Hard EM results (alpha, beta, placements)
        branch_proportions.tsv      Relative branch mutation burdens (pi_b)
        soft_em_results.pkl         Full soft EM results (pi_b, pi_0, history)

The hard EM output contains:
    - alpha         Estimated false-positive rate
    - beta          Estimated false-negative/dropout rate
    - placements    Hard MAP assignment of each SNV to a tree node

The soft EM output (branch_proportions.tsv) contains:
    - node          Node/branch name (matches tree and downstream tools)
    - pi_b          Relative mutation burden (sums to 1 across all branches)
    - is_leaf       Whether the node is a leaf cell


3) Subtree Refinement

Refines CNA-identical clades using SNV placements from the EM stage.

    sntree refine <sample> <input_root> <output_root>

Behavior:
    - If EM results exist, refinement uses EM placements.
    - Otherwise, if ML results exist, refinement uses ML placements.
    - If neither exists, an error is raised.

Outputs are written to:

    output_root/<sample>/sntree/refine/
        refined_full_tree.new
        group_XXX/
            refined_subtree.newick
            snv_assignments.tsv
            snv_assignments_final.tsv
            likelihood.txt


4) Full Pipeline

Runs preprocess -> EM (including soft branch-proportion EM) ->
subtree refinement sequentially.

    sntree pipeline <sample> <input_root> <output_root>

Note:
    The pipeline runs EM (not ML) before refinement.
    The soft branch-proportion EM runs automatically within the EM stage.
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
        --nni-max-iters 100 \
        --em-max-iters 50 \
        --soft-em-max-iters 50 \
        --alpha-dir 1.0

Soft EM-specific flags:

    --soft-em-max-iters INT     Maximum soft EM iterations (default: 50)
    --alpha-dir FLOAT           Dirichlet concentration on pi_b (default: 1.0).
                                Values < 1 encourage sparse branch attribution;
                                values > 1 push toward uniform.
    --soft-em-joint             Also update alpha/beta during the soft EM pass
                                (default: off; alpha/beta fixed from hard EM).
    --no-soft-em                Skip the soft EM phase entirely.

You may also override individual input file paths. Explicit paths take
precedence over paths derived from <input_root>/<sample>:

    sntree pipeline C2 /input /output \
        --medicc-tree /path/to/C2_final_tree.new \
        --cna-profiles /path/to/C2_final_cn_profiles.tsv \
        --cna-distances /path/to/C2_pairwise_distances.tsv \
        --sample-mapping /path/to/C2.info.tsv \
        --vcf /path/to/consensus_singlecell_counts.vcf.gz

If all required input files are provided explicitly, input_root can be any
placeholder directory. The sample argument is still used for output naming.


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
        consensus/
            final_singlecell_counts_merged.snvs.vcf.gz
    normal_cells/
        <sample>_normal_markdup.bam


------------------------------------------------------------
Output Directory Structure
------------------------------------------------------------

<output_root>/<sample>/sntree/
    ml/
    em/
        placements_em.tsv
        em_results.pkl
        branch_proportions.tsv      (soft EM output)
        soft_em_results.pkl         (soft EM output)
    refine/

Each stage writes only its own outputs and can be rerun independently.


------------------------------------------------------------
Typical Usage
------------------------------------------------------------

Full workflow (recommended):

    sntree pipeline C2 /path/to/input /path/to/output

With soft EM options:

    sntree pipeline C2 /input /output \
        --soft-em-max-iters 50 \
        --alpha-dir 1.0

Skip soft EM (original behaviour):

    sntree pipeline C2 /input /output --no-soft-em

Manual staged workflow:

    sntree preprocess C2 /input /output
    sntree em C2 /input /output
    sntree refine C2 /input /output


------------------------------------------------------------
Soft Branch-Proportion Inference
------------------------------------------------------------

The soft EM extends the hard SNV calling framework to estimate relative
branch mutation burdens (pi_b) without making hard variant calls at
individual loci. Every candidate site contributes to branch attribution
weighted by how well its read-count pattern matches each branch's
expected VAF fingerprint, determined by the CNA history and cell fractions.

Key properties:

    - pi_b sums to 1 across all branches and is interpretable as the
      fraction of all somatic mutations that arose on each branch.

    - Under a molecular clock, pi_b is proportional to elapsed
      evolutionary time on each branch.

    - Sub-threshold loci (which would be discarded by hard calling) still
      contribute soft evidence, improving estimates on small clades and
      tip branches.

    - The identifiability of pi_b within CNA-identical clades depends on
      the per-cell read distribution rather than predicted VAF alone.
      For highly clonal samples, running a second soft EM pass after
      subtree refinement improves within-clade branch length estimates.

    - pi_b can be clock-corrected downstream using SBS5 signature
      fractions: pi_b_clock = pi_b * SBS5_frac_b, normalised to sum to 1.
      See sandbox/sbs5_branch_lengths_plan.md for the recommended workflow.


------------------------------------------------------------
Notes
------------------------------------------------------------

- The refinement stage assumes CNA-identical groups have no CN transitions
  below their MRCA.
- The EM stage uses the CNA-aware inside-outside likelihood model.
- The soft EM reuses inside-outside scores already computed during the hard
  EM; it adds negligible runtime beyond the hard EM pass.
- The refinement stage uses a constant-CN cached likelihood for efficiency.
- ML and EM are alternative placement engines; refinement consumes either.
- All stages are independent and resumable.

End of README.
