------------------------------------------------------------
sntree: Single-Cell SNV Phylogenetic Inference and Refinement
------------------------------------------------------------

sntree performs:

1) CNA-aware maximum likelihood SNV placement
2) Soft branch-proportion inference (relative mutation burden pi_b per branch)
3) Local SNV-based refinement of CNA-identical subtrees
4) Optionally: hard EM estimation of sequencing error parameters (alpha, beta)

The default pipeline is soft-EM-first: alpha/beta are estimated jointly
with pi_b in pass 1, without ever making hard variant calls. Hard EM is
available as an opt-in for users who need MAP placements independently.


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

1) sntree preprocess

Preprocesses the MEDICC2 tree: normalises branch lengths and identifies
CNA-identical clades for downstream refinement.

    sntree preprocess <sample> <input_root> <output_root>

Output: output_root/<sample>/sntree/tree_preprocessed.new


2) sntree soft-em

Soft branch-proportion EM. Estimates pi_b (relative mutation burden per
branch) and alpha/beta using all candidate sites without hard calls.
Reuses the inside-outside DP scores from the CNA-aware likelihood model.

    sntree soft-em <sample> <input_root> <output_root>

Run pass 2 after refinement (warm start):

    sntree soft-em <sample> <input_root> <output_root> \
        --refined-tree <output_root>/<sample>/sntree/refine/refined_full_tree.new \
        --warm-start-pkl <output_root>/<sample>/sntree/soft_em/soft_em_results.pkl

Options:
    --refined-tree PATH      Use a refined tree instead of the preprocessed one.
    --warm-start-pkl PATH    Load pi_b and alpha/beta from a previous run
                             (warm-start for pass 2 after refinement).
    --soft-em-joint          Force joint estimation of alpha/beta even if hard
                             EM results already exist.
    --output-subdir NAME     Override the output subdirectory name
                             (default: soft_em, or soft_em_pass2 when
                             --refined-tree or --warm-start-pkl is given).

Outputs (written to output_root/<sample>/sntree/soft_em/ by default):
    branch_proportions.tsv      pi_b per branch, sorted by magnitude
    placements_soft.tsv         MAP SNV placements derived from soft EM,
                                with per-SNV confidence (see below)
    soft_em_results.pkl         Full results: pi_b, pi_0, alpha, beta, history,
                                and the scoring conditions (p0, p1_fp_mode,
                                tree_path) used by 'sntree score'

placements_soft.tsv columns:
    snv             SNV id
    node            MAP placement, or "Null"
    llr_null        best branch score minus the null score — how much better
                    the best branch is than "this is noise".  Null placements
                    report it as computed (<= 0).
    llr_margin      best minus second-best branch score — how confident the
                    branch choice is
    score_status    ok               both LLRs finite; only these loci are
                                     eligible for LLR-based filtering
                    no_margin        fewer than two branches possible;
                                     llr_margin is NaN
                    no_valid_branch  every branch impossible; llr_null is -inf
                    null_impossible  the null is impossible; llr_null is +inf
                    undefined        neither is possible; llr_null is NaN

The snv and node columns are unchanged from earlier versions.  Non-finite
values are written raw rather than clamped, so readers must handle them: in R
use read.delim(..., na.strings = c("NA", "NaN")) and expect lowercase "inf" /
"-inf" in rows whose score_status is not "ok".


3) sntree score

Scores placements from converged soft EM parameters — one E-step pass rather
than a full EM run — and optionally builds an empirical null for llr_null by
permuting the assignment of cells to tree tips.

    sntree score <sample> <input_root> <output_root>
    sntree score <sample> <input_root> <output_root> \
        --subsample-loci 200000 --permute --replicates 2

Why permute: a real mutation's alt-carrying cells sit in one clade, so
shuffling which cell sits at which tip destroys their concordance and its
llr_null collapses.  A site-recurrent artefact is scattered to begin with and
is unaffected.  The permuted pass therefore gives a null distribution for
llr_null against which a threshold can be calibrated empirically.  Each cell's
read counts move with it intact; only the cell → tip mapping changes, and a
fresh shuffle is drawn per locus batch so the resulting null loci are
near-independent.

The null is over loci, not replicates: one permuted pass over 200k loci is
already enough to place a threshold quantile.  Use --replicates 2 or 3 only to
check stability.

Options:
    --results-pkl PATH       soft_em_results.pkl to score
                             (default: .../sntree/soft_em/soft_em_results.pkl)
    --refined-tree PATH      Tree the parameters were fit on.  Required when
                             scoring a pass-2 fit; the run aborts rather than
                             score under a mismatched prior.
    --output-subdir NAME     Override the output subdirectory
                             (default: the directory holding the results pkl)
    --permute                Also run permuted passes
    --seed INT               Permutation seed (default 0); replicate r uses seed+r
    --replicates INT         Number of permuted passes (default 1)
    --subsample-loci N       Score only N randomly chosen loci (default: all)
    --subsample-seed INT     Seed for locus subsampling (default 0).  Separate
                             from --seed, so the observed pass and every
                             replicate score exactly the same loci.

Outputs (alongside the scored parameters, e.g. .../sntree/soft_em/):
    llr_observed.tsv            Per-SNV scores, columns as placements_soft.tsv
    llr_permuted_seed<S>.tsv    Same, one per permuted replicate
    llr_status_counts.tsv       score_status counts per run
    llr_summary.tsv             llr_null quantiles, observed against the
                                permuted null, and how many observed loci
                                survive each permuted quantile as a threshold

Quantiles are computed over score_status == "ok" loci only, in both the
observed and the permuted runs — including non-ok loci in one but not the
other would make the two distributions non-comparable and the threshold wrong.

llr_summary.tsv is stratified by the layer each locus was placed in (truncal =
the root, subclonal = any other branch, null, and pooled "all").  This matters:
a truncal placement barely moves under tip permutation, because the root's
clade is every cell and shuffling which cell sits at which tip changes almost
nothing.  Only subclonal placements lose their concordance.  When the truncal
layer dominates — as it does at low coverage — the tail of a pooled null is
made up entirely of loci the permutation could not touch, and a threshold read
off it is far too high for the subclonal layer it was meant to filter.  Take
the subclonal threshold from the "subclonal" rows.


4) sntree em  [hard EM — opt-in]

Hard EM: iteratively estimates alpha and beta and produces MAP SNV
placements via argmax. Use this if you specifically need hard calls
or want to supply alpha/beta to downstream tools independently.

    sntree em <sample> <input_root> <output_root>

Outputs (output_root/<sample>/sntree/em/):
    placements_em.tsv           Hard MAP SNV placements
    placements_em_loglik.tsv    Per-SNV log-likelihoods
    em_results.pkl              alpha, beta, placements, history


5) sntree refine

Refines CNA-identical clades using SNV placements from a prior stage.

    sntree refine <sample> <input_root> <output_root>

Behavior:
    - Looks for soft EM results (soft_em/soft_em_results.pkl) first and uses
      the MAP placements derived from soft EM.
    - Falls back to hard EM results (em/em_results.pkl) if soft EM was not run.
    - Raises an error if neither exists.

Outputs (output_root/<sample>/sntree/refine/):
    refined_full_tree.new
    group_XXX/
        refined_subtree.newick
        snv_assignments.tsv
        snv_assignments_final.tsv
        likelihood.txt


6) sntree pipeline

Full pipeline, run end-to-end.

Default (soft EM only):
    preprocess → soft EM pass 1 (joint alpha/beta) → refine → soft EM pass 2

With hard EM before soft EM:
    preprocess → hard EM → soft EM pass 1 (fixed alpha/beta) → refine → soft EM pass 2

    sntree pipeline <sample> <input_root> <output_root>

Options:
    --hard-em               Run hard EM before soft EM pass 1. Provides
                            independent MAP placements and alpha/beta estimates.
                            By default, soft EM jointly estimates alpha/beta.
    --no-soft-em-pass2      Skip soft EM pass 2 (no second run after refinement).
    --no-refine             Run only soft EM pass 1; skip refinement and pass 2.


------------------------------------------------------------
Typical Usage
------------------------------------------------------------

Full pipeline (recommended, soft EM default):

    sntree pipeline C2 /path/to/input /path/to/output

Full pipeline with hard EM (for MAP placements + independent alpha/beta):

    sntree pipeline C2 /input /output --hard-em

Manual staged workflow (soft EM default):

    sntree preprocess C2 /input /output
    sntree soft-em    C2 /input /output
    sntree refine     C2 /input /output
    sntree soft-em    C2 /input /output \
        --refined-tree /output/C2/sntree/refine/refined_full_tree.new \
        --warm-start-pkl /output/C2/sntree/soft_em/soft_em_results.pkl

Soft EM pass 1 only (skip refinement and pass 2):

    sntree pipeline C2 /input /output --no-refine

Calibrating a per-SNV confidence threshold after a pipeline run (the permuted
pass gives the null distribution llr_null should be thresholded against):

    sntree score C2 /input /output \
        --results-pkl /output/C2/sntree/soft_em_pass2/soft_em_results.pkl \
        --refined-tree /output/C2/sntree/refine/refined_full_tree.new \
        --subsample-loci 200000 --permute --replicates 2

Then read llr_summary.tsv: a permuted quantile is the candidate threshold, and
the observed counts beside it say how many placements survive it.

Hard MAP placements only (original behaviour):

    sntree preprocess C2 /input /output
    sntree em         C2 /input /output
    sntree refine     C2 /input /output


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

Soft EM parameters:

    --soft-em-max-iters INT     Maximum soft EM iterations per pass (default: 50)
    --alpha-dir FLOAT           Dirichlet concentration on pi_b (default: 1.0).
                                Values < 1 encourage sparse branch attribution;
                                values > 1 push toward uniform.

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
        consensus_singlecell_counts.vcf.gz
    normal_cells/
        <sample>_normal_markdup.bam


------------------------------------------------------------
Output Directory Structure
------------------------------------------------------------

<output_root>/<sample>/sntree/
    tree_preprocessed.new
    soft_em/
        branch_proportions.tsv      Relative mutation burden pi_b per branch
        placements_soft.tsv         MAP placements + per-SNV llr_null/llr_margin
        soft_em_results.pkl
        llr_observed.tsv            (if sntree score was run)
        llr_permuted_seed<S>.tsv    (if sntree score --permute was run)
        llr_status_counts.tsv
        llr_summary.tsv
    refine/
        refined_full_tree.new
        group_XXX/
    soft_em_pass2/                  (if pass 2 ran)
        branch_proportions.tsv
        placements_soft.tsv
        soft_em_results.pkl
    em/                             (only if --hard-em or sntree em was run)
        placements_em.tsv
        placements_em_loglik.tsv
        em_results.pkl

Each stage writes only its own outputs and can be rerun independently.


------------------------------------------------------------
Soft Branch-Proportion Inference
------------------------------------------------------------

The soft EM estimates relative branch mutation burdens (pi_b) without
making hard variant calls. Every candidate site contributes to branch
attribution weighted by how well its read-count pattern matches each
branch's expected VAF fingerprint, determined by the CNA history and
cell fractions.

Key properties:

    - pi_b sums to 1 across all branches and is interpretable as the
      fraction of all somatic mutations that arose on each branch.

    - Under a molecular clock, pi_b is proportional to elapsed
      evolutionary time on each branch.

    - Sub-threshold loci (which would be discarded by hard calling) still
      contribute soft evidence, improving estimates on small clades and
      tip branches.

Two-pass workflow:
    Pass 1 runs on the MEDICC2-derived preprocessed tree. It jointly
    estimates alpha/beta alongside pi_b (no hard EM needed). MAP
    placements are derived by argmax from the converged soft assignments
    and fed into subtree refinement.

    Pass 2 runs on the NNI-refined tree with pi_b warm-started from
    pass 1. Branch proportions are matched by node name across the two
    topologies; unmatched nodes (new within-clade branches from NNI)
    are initialised to the mean of matched values. Alpha/beta are fixed
    from pass 1.

CNA-identical clades:
    Within these clades, predicted VAF is identical for all branches,
    so only per-cell read distributions discriminate them. For clonal
    samples, the two-pass approach is especially important: NNI
    refinement resolves within-clade topology, and pass 2 then assigns
    mutation burden to the refined branches.

Clock correction (downstream):
    pi_b can be SBS5-corrected downstream:
        pi_b_clock = pi_b * SBS5_frac_b, normalised to sum to 1.
    This weights each branch by the SBS5 clock-like fraction estimated
    from SigProfiler. See sandbox/sbs5_branch_lengths_plan.md.


------------------------------------------------------------
Notes
------------------------------------------------------------

- The refinement stage assumes CNA-identical groups have no CN transitions
  below their MRCA.
- The EM stage uses the CNA-aware inside-outside likelihood model.
- The soft EM reuses inside-outside DP scores; it adds negligible runtime
  compared to the inside-outside pass itself.
- The refinement stage uses a constant-CN cached likelihood for efficiency.
- All stages are independent and resumable.

End of README.
