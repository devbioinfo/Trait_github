# mlconv — calibrated multi-layer detection of convergent molecular evolution

Working name. Prototype for Manuscript 1 of the PhD project *A multi-omics study of
convergent evolution in selected zoological traits: an in silico approach*.

## The problem it solves

Convergence scans test whether a gene behaves differently on the branches of
lineages that independently evolved a trait (the *foreground*). Two problems make
published scans unreliable:

1. **Phylogenetic confounding.** Trait lineages are clumped on the tree, and a single
   ancient shift in one clade looks like a foreground effect. Tests that treat
   branches as independent are anticonservative (in our simulations, 17-34% false
   positives at a nominal 5%).
2. **Combining layers.** Evidence comes from several correlated layers (relative rates,
   convergent substitutions, selection, non-coding acceleration, functional effect).
   Fisher's method assumes independence and inflates false discoveries; no existing
   method reports *which* layers converge.

## Model

### 1. Phylogenetic linear mixed model per gene and layer

For gene *g*, layer *l*, the vector of branch values **y** (length *B*) is

    y = b0·1 + β·f + u + e,     u ~ N(0, h²K),   e ~ N(0, s²I)

* **f** is the foreground indicator (branches whose descendant tips all have the trait).
* **K**[b, b'] is the length of the root-to-branch path shared by branches b and b':
  the covariance of a Brownian process accumulated along the tree. It absorbs
  clade-level shifts that would otherwise masquerade as convergence.
* With δ = h²/s², V = s²(δK + I). One eigendecomposition K = U Λ Uᵀ, shared by all
  genes, turns every fit into weighted least squares with weights 1/(δλ_b + 1).
  REML over a δ grid is therefore vectorised across the genome (FaST-LMM idea).
* β is tested with a Wald *t* (df = B − 2), mapped to a one-sided z-score z_gl.

### 2. Cross-layer dependence

Whitened residuals r_gl = V^{-1/2}(y − Xβ̂) give a per-gene cross-layer correlation
R_g, shrunk toward the genome-wide mean. Under H0, z_g ~ N(0, R_g).

### 3. Whitened combination (WC)

    S_g = 1ᵀ R_g⁻¹ z_g / sqrt(1ᵀ R_g⁻¹ 1)  ~ N(0, 1) under H0

This is optimal when all layers shift together.

### 4. Empirical-Bayes configuration model (EBCM)

Convergence usually affects only some layers. Let γ ∈ {0,1}^L mark the active layers
and τ_k be a grid of effect scales:

    z_g ~ π₀ N(0, R_g) + Σ_{γ≠0} Σ_k π_{γ,k} N(0, R_g + τ_k² diag(γ))

* The weights π are learned genome-wide by EM, so the data decide which layer
  combinations are typical of convergence (e.g. *rate + selection* vs *non-coding only*).
* Outputs per gene: local FDR (lfdr_g = P(γ = 0 | z_g)), posterior activity of each
  layer, and the most probable configuration. Together these form the
  **convergence signature** of Objective 3.
* Convergence is one-sided (positive shift); components are truncated to that side.

### 5. Calibration by control lineages

Bayesian lfdr is optimistic when the model is misspecified. We therefore re-fit the
LMM on *M* structure-matched null foregrounds drawn only from lineages without the
trait (same number of origins, clade sizes within ±1 tip; the control-lineage logic of
Thomas & Hahn 2015). The observed lfdr (or S) is compared with the null pooled over
genes × draws (G·M values), and Benjamini–Hochberg is applied to the resulting
p-values.

### 6. Lineage-wide shifts

Before fitting, each layer is transformed to relative values (per branch, the
genome-wide median is subtracted), as relative evolutionary rates do in RERconverge.

## Code

| File | Contents |
|---|---|
| `mlconv/tree.py` | Random trees, foreground mapping, BM permulations, control-lineage clade nulls |
| `mlconv/simulate.py` | Branch-level multi-layer simulator with phylogenetic confounding and misspecification |
| `mlconv/lmm.py` | Genome-wide phylogenetic LMM (REML on an eigenbasis), cross-layer correlation, WC |
| `mlconv/configs.py` | EBCM: EM, lfdr, layer posteriors, Bayesian FDR |
| `mlconv/stats.py` | Baselines (naive, Fisher, ACAT, permulation), pooled-null calibration |
| `scripts/benchmark.py` | v0/v1 benchmark (permulation-only methods) |
| `scripts/benchmark_v2.py` | v2 benchmark (model-based methods, misspecification scenarios) |

Run: `python3 -I scripts/benchmark_v2.py --out results/benchmark_v2.csv`

## Development log (what the simulations showed so far)

1. Naive per-branch tests: 13-17% false positives under confounding; naive Fisher: 25-34%.
2. Permulations alone (BM or clade-matched): calibrated only on large trees. On a
   100-tip tree each gene's null is built from too few distinct clades (per-layer FPR
   0.08 at 100 tips, 0.06 at 200, 0.055 at 400), and combining layers amplifies the
   error. Conclusion: permulations work as a calibration layer, not as the core test.
3. Phylogenetic LMM: per-layer FPR ≈ 0.045 even under strong confounding.
4. Fisher on calibrated layers stays anticonservative (FDR 0.33-0.42 at target 0.10).
5. EBCM: highest power in sparse settings, layer attribution AUC 0.79-0.99, but raw
   Bayesian FDR optimistic → calibrated against the control-lineage null.

## Benchmark v2 results (3 replicates, 100-tip trees, 2,000 genes, 5% convergent)

Figure: `results/fig_benchmark.png`; raw rows: `results/benchmark_v2.csv`.

| Method | FPR at α = 0.05 (all scenarios) | Realised FDR at q = 0.10 | Power, effect 0.8, confounding 1 |
|---|---|---|---|
| Naive single layer | 0.12-0.16 under confounding | 0.63-0.82 | (invalid) |
| Naive Fisher | 0.24-0.31 | 0.77-0.86 | (invalid) |
| Permulation Fisher | 0.12-0.18 | 0.50-0.70 | (invalid) |
| Phylo-LMM Fisher | 0.10-0.11 | 0.29-0.54 | (invalid) |
| Phylo-LMM ACAT | 0.06-0.07 | 0.07-0.27 (fails: sparse 0.27, heavy tails 0.18) | 0.67 |
| Phylo-LMM whitened sum | 0.049-0.056 | 0.03-0.10 | 0.47 |
| **Phylo-LMM + EBCM** | **0.049-0.059** | **0.06-0.12** | **0.74** |

* EBCM is the only method that keeps FDR near target in every scenario while
  having the most power among calibrated methods in 5 of 7 scenarios.
* Weakness: when *all* layers shift together (dense), the whitened sum is more
  powerful (0.86 vs 0.65). An omnibus of EBCM and WC is the obvious next step.
* Slight FDR overshoot at strong confounding (0.124 at q = 0.10); needs more
  replicates and larger trees to judge.
* Layer attribution of detected genes: AUC 0.79-0.98.

## Sequence-level validation with IQ-TREE AliSim (in progress)

`scripts/alisim_validation.py` simulates codon alignments (GY model) with AliSim on
Yule trees. Convergent genes get, on foreground branches only, a higher ω
(branch-specific `GY{ω,κ}` models), faster evolution (longer branches), or both.
Confounding is added as gene-specific rate jumps on random clades. Branch lengths are
then re-estimated with IQ-TREE on the fixed topology (GTR on nucleotides, LG on the
translated protein). Layers: log nucleotide rate, log protein rate, and their
difference (a dN/dS-like selection proxy), with a one-substitution pseudo-count and
mean-centring per branch.

Findings so far (80-200 species, 600-3,000 genes):

1. **The Brownian phylogenetic LMM is miscalibrated on sequence-derived rates**
   (per-layer FPR 0.10-0.14 at α = 0.05). Causes: unequal estimation noise across
   branches (short branches are much noisier), correlated estimation errors between
   neighbouring branches, and rate *jumps* rather than Brownian drift. Per-branch
   weighting and a gene × branch measurement-error model did not fix it.
2. **Control-lineage nulls must match clade age, not just size.** Young trait clades
   consist of short, noisy branches; size-matched controls were three times longer
   (median 0.039 vs 0.129), so the null understated the variance.
3. **A covariance learned across genes gives exact calibration.** Because all genes
   share one tree, the B × B covariance of branch values estimated from the genome
   absorbs phylogeny, rate jumps, unequal noise and correlated errors at once.
   GLS with this covariance: FPR 0.038-0.053 for the observed foreground and z-score
   SD 0.99-1.03 for every control set (`mlconv/gls.py`).
4. **Contamination:** truly convergent genes inflate the learned covariance along
   the foreground direction, making the test conservative. Hard trimming and an EM
   soft-removal both became anticonservative; a rank-one correction estimated from
   excess covariance between *different* origins works for the selection layer but
   is noisy. The plain learned covariance stays valid (conservative).
5. **Single-lineage rate jumps** produce excess false discoveries in the raw-rate
   layers even with the true covariance; the selection proxy is immune. Requiring
   the signal to survive the removal of any single origin (leave-one-origin-out
   intersection-union test, or Benjamini-Heller partial conjunction) removes these
   false discoveries completely; its power depends on the number and size of origins.

### Realistic design (200 species, 5 origins of 4-12 species, 3,000 genes, 2% convergent, 2 trees)

Raw rows: `results/alisim_realistic.csv`. FPR at α = 0.05 on a genome with no convergence;
FDR and power at q = 0.10 on a genome with 2% convergent genes (about 60 per tree).

| Method | FPR (no convergence) | FDR | Power | Power: ω shift / rate shift / both |
|---|---|---|---|---|
| Naive, nucleotide rate | 0.177 | 0.857 | 0.68 | 0.20 / 1.00 / 0.95 |
| Naive, Fisher over layers | 0.195 | 0.826 | 1.00 | (invalid) |
| Brownian LMM, ACAT over layers | 0.064 | 0.310 | 0.98 | 1.00 / 0.95 / 1.00 |
| Brownian LMM + EBCM | 0.064 | 0.163 | 0.53 | 0.49 / 0.10 / 0.91 |
| Learned-covariance GLS, ACAT over layers | 0.051 | 0.122 | 0.97 | 1.00 / 0.89 / 1.00 |
| **Learned-covariance GLS, leave-one-origin-out, ACAT** | **0.003** | **0.000** | **0.93** | **1.00 / 0.76 / 1.00** |

* The leave-one-origin-out version made no false discoveries in either tree while
  finding 93% of convergent genes. With five origins, requiring the signal to survive
  the removal of any one lineage costs little power.
* Layers behave as expected biologically: the selection proxy detects every ω-shift
  gene and no pure rate-shift gene; the nucleotide rate layer does the opposite.
* EBCM on Brownian-LMM z-scores loses power on sequence data and is not the default;
  it should be re-fitted on the GLS z-scores.

### Power vs number of independent origins

`scripts/origins_power.py`; figure `results/fig_origins_power.png`; rows
`results/origins_power.csv`. 3 Yule trees of 200 species, 3,000 genes, 2% convergent,
nested trait configurations with 2-6 origins (clades of 4-12 species) on the same tree
and null genes.

| Origins | Final test (GLS + leave-one-origin-out, ACAT): power | FDR (q = 0.10) | False calls per null genome | Power: ω shift / rate only / both |
|---|---|---|---|---|
| 2 | 0.63 | 0.010 | 0.3 | 0.87 / 0.17 / 0.93 |
| 3 | 0.81 | 0.000 | 0.0 | 0.98 / 0.44 / 0.98 |
| 4 | 0.89 | 0.000 | 0.0 | 0.99 / 0.62 / 1.00 |
| 5 | 0.89 | 0.006 | 0.3 | 0.99 / 0.70 / 1.00 |
| 6 | 0.95 | 0.006 | 0.3 | 1.00 / 0.82 / 1.00 |

Comparison at the same settings: naive single-layer test FDR 0.83-0.88 (about 220-280
false calls per null genome); learned-covariance GLS without leave-one-origin-out FDR
0.26 (2 origins) falling to 0.10 (6 origins), about 20 false calls per null genome.

* The final test controls false discoveries at every number of origins, including 2.
* Changes in selection pressure (ω) are found at 87-100% power even with 2 origins.
* Rate-only convergence needs more origins: 17% (2), 44% (3), 62-82% (4-6).
* Implication: echolocation (2-4 origins) is feasible for selection-type convergence;
  aquatic and high-altitude adaptation (more origins) give power for rate shifts too.

### Convergence signatures: which layers converged

Detection uses the final test above. To describe *how* each detected gene converged,
`mlconv/signatures.py` fits a mash-style mixture (Urbut et al. 2019) to the per-layer
GLS z-scores: canonical on/off patterns plus directions learned from the strongest
genes, effect scales up to z = 30, weights by EM. Each gene is assigned its most
probable pattern; a learned direction marks a layer active when its loading exceeds 0.3.
The number of learned directions is fixed in advance at 2^L - 1 (7 for three layers);
BIC under-selects because about 98% of genes are null.

Evaluation (`scripts/signature_eval.py`, output `results/signatures/evaluation.txt`):
200 species, 5 origins, 3,000 genes, 2% convergent. Tree B was generated after the
rule was fixed and was not used for any tuning.

| Method | Tree A | Tree B (blind) |
|---|---|---|
| **Learned directions (k = 2^L - 1)** | **0.90** | **0.98** |
| Per-layer rule, p < 0.05 | 0.78 | 0.86 |
| EBCM (on/off configurations, wide effect grid) | 0.80 | 0.42 |

Exact accuracy = share of correctly detected convergent genes whose predicted set of
active layers equals the set implied by the simulation (selection shift: protein rate +
selection proxy; rate shift: nucleotide + protein rate; both: all three). Detection on
both trees: 92 true, 0 false.

Without being told the simulation, both trees learn the same three directions
(nucleotide, protein, selection loadings): rate shift ≈ (0.84, 0.55, 0.0), selection
shift ≈ (0.1, 0.55, 0.82), both ≈ (0.5, 0.65, 0.6). EBCM is unstable across trees and
is no longer the default signature method.

## Manuscript benchmark (Figure 2)

`scripts/final_benchmark.py`; figure `results/fig_final_benchmark.png`; rows
`results/final_benchmark.csv`. AliSim codon data, 3 Yule trees of 200 species, 5 origins
(4-12 species each), 3,000 genes, 2% convergent; confounding 0/1/2; effect sizes weak
(ω x1.5, rate x1.2), moderate (x2, x1.35), strong (x3, x1.5). Paired design: per tree and
confounding level one null genome, convergent genes swapped in per effect size.

False calls on a genome with no convergence (BH q = 0.10, mean per genome) and realised
FDR with 2% convergent genes:

| Method | False calls, conf 0 / 1 / 2 | FDR, conf 0 / 1 / 2 | Power, weak / moderate / strong |
|---|---|---|---|
| Naive (RERconverge-style) | 0.3 / 216 / 401 | 0.08 / 0.83 / 0.90 | 0.44 / 0.63 / 0.66 |
| Control-lineage permutation | 182 / 228 / 239 | 0.77 / 0.84 / 0.88 | 0.44 / 0.57 / 0.64 |
| Naive, Fisher over layers | 13 / 311 / 497 | 0.25 / 0.82 / 0.88 | 0.88 / 0.97 / 0.99 |
| Brownian phylo-LMM | 0 / 10 / 25 | 0.04 / 0.31 / 0.40 | 0.76 / 0.93 / 0.98 |
| Learned-covariance GLS | 0 / 17 / 13 | 0.00 / 0.19 / 0.19 | 0.75 / 0.91 / 0.95 |
| **GLS + leave-one-origin-out (final)** | **0 / 0 / 0.7** | **0.00 / 0.00 / 0.01** | **0.56 / 0.84 / 0.90** |

* Only the final test keeps false discoveries below target at every confounding level;
  it made at most 2 false calls in any of the 9 null genomes (most had none).
* Power of the final test falls with confounding (0.89 / 0.74 / 0.67 at conf 0 / 1 / 2);
  that is the price of rejecting single-lineage rate jumps.
* By kind (final test, weak / moderate / strong): selection shift 0.56 / 0.95 / 1.00;
  selection + rate 0.80 / 0.97 / 1.00; rate shift only 0.24 / 0.57 / 0.62.
* Signature accuracy (exact pattern of active layers, detected genes): 0.76 / 0.87 / 0.94
  (range over the 9 settings 0.47-0.90 / 0.53-0.98 / 0.88-0.98).
* "Control-lineage permutation" is our implementation (Welch t standardised by 100
  age-matched control foregrounds), not the RERconverge package; it is inflated even
  without confounding because per-gene null moments come from a limited pool of clades.
  The manuscript should also compare against RERconverge itself.

## Head-to-head comparison with RERconverge

`scripts/rerconverge_compare.py` + `scripts/run_rerconverge.R`; rows
`results/rerconverge_compare.csv`. RERconverge 0.3.0 (github.com/nclark-lab/RERconverge,
commit 682ac34) with `scripts/rerconverge_no_ggtree.patch` (ggtree made optional; it is
used only for plotting). Settings: `getAllResiduals(transform = "sqrt", norm = "scale")`,
`foreground2Tree(clade = "all")`, `correlateWithBinaryPhenotype(weighted = "auto")`;
binary permulations (`getPermsBinary`, permmode "cc", inputs from
`getBinaryPermulationInputsFromTree`) with 1,000 permulations on the 150 genes with the
smallest parametric P (other genes treated as not significant). Same AliSim genomes as the
final pipeline: 1 Yule tree of 200 species, 5 origins, 3,000 genes; confounding 0 / 1 / 2;
a null genome and a genome with 2% convergent genes (moderate effect).

| Method | False calls, null genome (conf 0 / 1 / 2) | FDR, 2% convergent (target 0.10) | Power |
|---|---|---|---|
| RERconverge, nucleotide trees | 0 / 37 / 119 | 0.07 / 0.54 / 0.75 | 0.65 / 0.67 / 0.60 |
| RERconverge, protein trees | 0 / 28 / 78 | 0.00 / 0.43 / 0.61 | 0.93 / 0.87 / 0.82 |
| RERconverge, protein + permulations | 0 / 0 / 0 | 0.39 / 0.39 / 0.33 | 0.93 / 0.72 / 0.60 |
| **Final pipeline** | **0 / 0 / 0** | **0.00 / 0.00 / 0.00** | **0.87 / 0.72 / 0.65** |

* Without lineage-specific rate jumps (confounding 0), RERconverge on protein trees is
  calibrated and slightly more powerful than the final pipeline (0.93 vs 0.87).
* With rate jumps, parametric RERconverge loses FDR control (0.43-0.61).
* Permulations removed all false calls on the null genomes, but on the convergent genomes
  18-35 null genes were called (FDR 0.33-0.39). This needs checking before it is
  reported (see caveats): it may reflect the candidate-only permulation design rather
  than permulations themselves.
* The final pipeline made no false calls in any of the six genomes.

Caveats: one tree only; permulations restricted to 150 candidates for run time
(about 1 s per permulation at 200 species on 4 cores).

## Roadmap to the manuscript

1. **Benchmark** (in progress): calibration and power across confounding,
   heavy tails, lineage shifts, tree error, sparse/dense signal.
2. **Sequence-level validation:** simulate codon alignments with IQ-TREE AliSim (rate
   shifts on foreground branches), compute real layer statistics, and confirm the
   results hold beyond the Gaussian branch-level simulator.
3. **Real layer extraction from Zoonomia / TOGA alignments:**
   relative rates (RERconverge residuals), convergent substitutions (CSUBST),
   branch-level selection (aBSREL), non-coding acceleration (PhyloAcc), and an
   **AI layer**: ancestral reconstruction plus ESM-2 protein language model scores of
   each branch's substitutions (predicted functional effect).
4. **Head-to-head comparison** with RERconverge, CSUBST, PCOC and a Marcovitz-style
   enrichment test.
5. **Application:** echolocation (bats and toothed whales vs pteropodids and baleen
   whales). Positive control: Prestin/SLC26A5. Negative control for *adaptive*
   convergence: eye-gene decay in subterranean mammals, which should appear as rate
   shifts without functional-effect convergence.
6. **Package:** pip-installable, documented, with Zenodo DOI.

## AI-use disclosure

Nature Portfolio journals do not accept AI tools as authors and require the use of
large language models to be documented in the Methods. Code written with AI
assistance must be reviewed, tested and owned by the authors.
