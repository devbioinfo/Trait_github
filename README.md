# mlconv — calibrated multi-layer detection of convergent molecular evolution

Working name. Prototype for Manuscript 1 of the PhD project *A multi-omics study of
convergent evolution in selected zoological traits: an in silico approach*.

## Final results (as in the manuscript)

Manuscript: *Convergent evolution in mammals is written in gene loss* (in preparation for
*Nature Ecology & Evolution*). Every number below is produced by the scripts in this
repository; sections further down are the development log that led here and are kept as a
record (numbers there may be superseded by this section).

**Data.** 115 mammals (Zoonomia 447-way tree, TOGA orthology against human hg38);
15,703 genes tested for rate shifts, 12,851 for gene loss. Traits fixed before testing:
marine (fully aquatic lineages of marine origin: cetaceans including two river dolphins,
pinnipeds, manatee; 3 origins), subterranean (6), high altitude (4), echolocation (3).

**Test.** Four layers per gene (nucleotide rate, protein rate, selection proxy, gene loss);
null covariance learned from all genes (GLS); every call must survive leave-one-origin-out;
gene-loss P values from a simulated irreversible-loss null with species x gene annotation
error; 20 matched control traits per analysis.

### Real data

| Trait | Origins | Rate calls | Loss calls |
|---|---|---|---|
| Marine | 3 | 0 | 48 (smell, taste, skin, detoxification; PON1) |
| Subterranean | 6 | 0 | 15 (lens and retina; LGSN strongest, P = 5 x 10^-9) |
| High altitude | 4 | 0 | 1 (DEAF1, lost in two species; probable annotation error) |
| Echolocation | 3 | 0 | 0 |

* Calibration on real genomes: 500 matched control traits (100 per analysis); any rate call in 3%
  (95% interval 2-5%), any loss call in 13% (10-16%), on average 0.08-0.22 false loss calls per
  control trait (0.66 for high altitude, one control trait with 33 calls in loss-rich genomes).
* Robustness of the loss calls: the same 48 marine and 15 subterranean genes with 500 or 1,000
  simulated null genomes and with the tail fitted to 100 or 500 values. Counting only TOGA 'lost'
  (uncertain losses as missing): 28 marine calls (17 shared), 26 subterranean (11 shared, the new ones
  mostly eye genes), DEAF1 no longer called, 41 exploratory echolocation calls
  (`results/audit/`, Supplementary Notes 12-13).
* Evidence per loss: 48 of 63 carry an inactivating mutation in every origin where they are lost
  (`results/extra/loss_evidence_by_gene.csv`); prior literature per gene in Supplementary Table 1.
* Enrichment against a loss-frequency-matched null: olfactory signalling 4.1x (q = 0.0002),
  sweet/bitter/umami taste 19x (q = 0.02) for marine losses; retina-elevated genes 8.8x
  (q = 0.008), lens development 92x (q = 0.0005) for subterranean losses.
* PON1: lost once in the cetacean ancestor (shared frameshifts and splice-site mutation),
  at least three times within pinnipeds, separately in the manatee. LGSN: independent
  frameshifts in golden mole, Damara mole-rat and blind mole-rat; the two moles share one.
* Underground, lens and image-processing genes were lost while the phototransduction core
  (RHO, GNAT1, PDE6A, CNGA1, PRPH2) stayed intact in all eight subterranean species.
* The combined hypoxia trait's 11 losses (nine eye genes) are explained by life
  underground (partial effect +0.62), not high altitude (-0.04).

### Simulations

| Benchmark | Result (final test) | Comparison |
|---|---|---|
| Rate layers, 10 AliSim trees of 200 species, 3,000 genes (Extended Data Fig. 1) | FDR 0.000 / 0.003 / 0.007 at rate-jump intensity 0 / 1 / 2; false calls in 2 of 30 null genomes (at most 2) | other methods 0.19-0.91 once rate jumps are present |
| Sensitivity: 100 or 400 species, 0.5-10% convergent, gene-tree discordance; 5 trees each (Extended Data Fig. 3) | mean FDR <= 0.005 (max 0.05), at most 1 false call per null genome; power 0.59-0.72, except 0.36 with 10% convergent genes | learned-covariance GLS alone 0.03-0.56 |
| Number of origins, 10 replicates (Extended Data Fig. 1e,f) | FDR <= 0.01 at 2-6 origins; selection shifts found at 0.72 (2 origins) to 0.99 (6) | rate-only shifts need >= 4 origins for power > 0.6 |
| Gene-loss layer, 5 replicate genomes per setting on the real tree (Fig. 3) | false discovery proportion 0.019 on average (max 0.067); power 0.29-0.89; 1 false call in 15 null genomes | Fisher test without phylogeny: up to 0.69 |
| RERconverge 0.3.0 (Extended Data Fig. 2) | final test: no false calls in any genome | parametric RERconverge FDR 0.43-0.75 with rate jumps; with 10,000 permulations on all genes FDR 0.24 / 0.04 / 0.15 at intensity 0 / 1 / 2 (one genome each; final test 0 false calls, power 0.97 / 0.83 / 0.72), at about 11 CPU-hours per genome |

Reproducibility: tree 3 of the main benchmark was run on two machines (Linux x86 and
macOS ARM); the final test gave identical false-call counts and FDR, and power within 0.05.

### Figures and how to rebuild them

`python3 -I scripts/make_figures.py` writes every figure to `results/figures/` and checks
it against the Nature Portfolio rules (at most 183 x 170 mm, text 5-7 pt, no overlapping
text). Main figures: 1, species tree and trait origins; 2, method; 3, gene-loss benchmark;
4, results per trait; 5, PON1; 6, eye genes. Extended Data Figs. 1-10;
`StudySummary` is a one-page overview. Silhouettes: game-icons.net (CC BY 3.0),
`scripts/silhouettes.json`.

The robustness runs (`results/robustness/`) are rerun with
`bash scripts/run_on_workstation.sh` (resumable; `bash scripts/check_workstation.sh` checks
the requirements, `bash scripts/install_rerconverge.sh` installs RERconverge 0.3.0 with
our patch). RERconverge with all-gene permulations:
`RER_CORES=20 python3 scripts/rerconverge_compare.py --confounding 1 --effects moderate --perms 10000 --candidates 0 --seed 5031 --out results/robustness/rerconverge_allperm10k.csv`.

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

## Development log (early simulations; superseded by "Final results" above)

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

## Sequence-level validation with IQ-TREE AliSim

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

## Manuscript benchmark (Extended Data Fig. 1)

`scripts/final_benchmark.py`; rows `results/final_benchmark.csv` (trees 0-2) and
`results/robustness/main_tree3.csv` ... `main_tree9.csv` (trees 3-9;
`main_tree3_mac.csv` is the same tree rerun on macOS). AliSim codon data, 10 Yule trees of
200 species, 5 origins (4-12 species each), 3,000 genes, 2% convergent; confounding 0/1/2;
effect sizes weak (ω x1.5, rate x1.2), moderate (x2, x1.35), strong (x3, x1.5). Paired
design: per tree and confounding level one null genome, convergent genes swapped in per
effect size.

False calls on a genome with no convergence (BH q = 0.10, mean per genome) and realised
FDR with 2% convergent genes, mean over 10 trees:

| Method | False calls, conf 0 / 1 / 2 | FDR, conf 0 / 1 / 2 | Power, weak / moderate / strong |
|---|---|---|---|
| Naive (RERconverge-style) | 0.9 / 210 / 395 | 0.15 / 0.84 / 0.91 | 0.43 / 0.60 / 0.64 |
| Control-lineage permutation | 124 / 189 / 199 | 0.73 / 0.84 / 0.86 | 0.42 / 0.54 / 0.60 |
| Naive, Fisher over layers | 63 / 327 / 504 | 0.42 / 0.83 / 0.89 | 0.86 / 0.98 / 0.99 |
| Brownian phylo-LMM | 6.2 / 20 / 39 | 0.08 / 0.36 / 0.45 | 0.67 / 0.93 / 0.98 |
| Learned-covariance GLS | 0 / 16 / 16 | 0.00 / 0.19 / 0.24 | 0.70 / 0.91 / 0.96 |
| **GLS + leave-one-origin-out (final)** | **0 / 0.1 / 0.2** | **0.000 / 0.003 / 0.007** | **0.49 / 0.81 / 0.91** |

* Only the final test keeps false discoveries below target at every confounding level;
  it made false calls in 2 of the 30 null genomes (at most 2 calls), and its FDR in any
  single genome was at most 0.06.
* Power of the final test falls with confounding (0.88 / 0.70 / 0.63 at conf 0 / 1 / 2);
  that is the price of rejecting single-lineage rate jumps.
* By kind (final test, weak / moderate / strong): selection shift 0.51 / 0.89 / 0.98;
  selection + rate 0.71 / 0.95 / 1.00; rate shift only 0.23 / 0.55 / 0.71.
* Signature accuracy (exact pattern of active layers, detected genes): 0.74 / 0.88 / 0.95.
* "Control-lineage permutation" is our implementation (Welch t standardised by 100
  age-matched control foregrounds), not the RERconverge package; it is inflated even
  without confounding because per-gene null moments come from a limited pool of clades.

Sensitivity runs (confounding 1, three trees each; `results/robustness/sens_*.csv`,
Extended Data Fig. 3): final-test FDR at most 0.01 and at most one false call per null
genome with 100 or 400 species, 0.5%, 5% or 10% convergent genes, and gene-tree
discordance (Poisson mean 10 or 40 NNIs per gene). Power: baseline 0.71; 100 species 0.67;
400 species 0.67; 0.5% convergent 0.67; 5% 0.56; 10% 0.36 (a large convergent signal
enters the covariance learned from all genes); discordance 10 or 40, 0.62.

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
* Permulations on the 150 candidate genes removed all false calls on the null genomes,
  but on the convergent genomes 18-35 null genes were called (FDR 0.33-0.39).
* **Permulations on all genes** (`results/robustness/rerconverge_allperm10k.csv`;
  10,000 permulations, confounding 1, a second genome of the same design): no false calls
  on the null genome, FDR 0.04 and power 0.83 with 2% convergent genes, the same power as
  the final pipeline on that genome (0.83, no false calls). The excess above therefore came
  from the candidate-only design. With 100 permulations
  (`rerconverge_allperm.csv`) no gene can be called: the smallest attainable P (about
  0.02) is above the BH threshold for 3,000 genes.
* Cost: about 11 CPU-hours per genome for all-gene permulations, against seconds for the
  final pipeline.
* The final pipeline made no false calls in any of the six genomes.

## Real-data pilot: echolocation on RERconverge's mammal gene trees

`scripts/echolocation_pilot.py` (+ `scripts/export_rerconverge_trees.R`,
`scripts/rer_foregrounds.R`); results in `results/echolocation_pilot/`. Data: the protein
gene trees bundled with RERconverge (`mamTreesSub`), 761 genes present in all 62 species,
exported to `data/rerconverge_mammals/`. Trait: echolocation in vespertilionid bats
(eptFus1, myoDav1, myoLuc2) and toothed whales (turTru2, orcOrc1): 2 origins, 8 foreground
branches; pteropodid bats are non-echolocating. One layer only (log protein branch length).

Real-data calibration: the same analysis on 100 control trait sets (two non-echolocating
clades matched to the echolocating origins in size and age; any call is false).

| Method | Mean false calls per control set | Control sets with any call | Max | Echolocation calls |
|---|---|---|---|---|
| Naive (branches independent) | 73.8 | 100% | 170 | 12 |
| RERconverge (parametric) | 0.85 | 22% | 17 | 0 |
| Learned-covariance GLS | 0.75 | 43% | 9 | 0 |
| **Final (GLS + leave-one-origin-out)** | **0** | **0%** | **0** | **0** |

* On real genomes the final test made no false calls on any of the 100 control trait sets.
* No gene is significant for echolocation with any calibrated method in this subset
  (the naive test's 12 calls are well inside its control distribution). Expected: 761
  genes, 5 echolocating species in 2 origins, one layer, and none of the classic
  echolocation genes except CDH23 (final p = 0.40).
* This pilot tests calibration on real data, not biology.

## Echolocation on Zoonomia / TOGA (genome-wide)

Data: TOGA multiple codon alignments (hg38 reference, 17,439 genes), 101 species chosen by
fixed rules before any test (`scripts/toga_select_species.py`: all well-covered bats and
cetaceans, a backbone from every other order, shrews/tenrecs/solenodons excluded);
per-gene branch lengths on the pruned Zoonomia tree with IQ-TREE 2 (GTR+G4, LG+G4;
`scripts/toga_branch_lengths.py`). 15,757 genes passed (>= 80% of species, >= 50 codons);
4.9% of branch values are missing. Trait: echolocating bats (Rhinolophoidea,
Yangochiroptera) and toothed whales, 25 species, 3 origins, 47 foreground branches.
Calibration: 20 control trait sets matched to the three origins in size and age
(`scripts/convergence_toga.py --trait echolocation`, results in `results/echolocation_toga/`).

| | Calls (q = 0.1) | Smallest p |
|---|---|---|
| Echolocation | 0 | 0.0014 |
| Control sets (20) | mean 0.2; 4 of 20 sets with one call | median 0.001 |

* No gene shows genome-wide convergence in rates for echolocation. Prestin/SLC26A5:
  p = 0.47. The 17 hearing genes present (MYO15A missing) are not enriched
  (Mann-Whitney p = 0.40); the best are TMC1 (0.026) and LOXHD1 (0.029).
* The echolocation top gene is weaker than the top gene of 14 of the 20 control sets.
* With three origins the leave-one-origin-out test rests on two origins at a time, where
  simulated power is lowest; the p-value distribution is skewed towards 1
  (conservative). Prestin's reported convergence is at specific amino-acid sites, which
  a whole-gene rate test is not designed to detect.
* Next: traits with more origins (secondary aquatic adaptation, hypoxia tolerance) and a
  site-level convergent-substitution layer.

## Marine adaptation on Zoonomia / TOGA (101 species, preliminary)

Same data and pipeline as the echolocation analysis (`scripts/convergence_toga.py --trait marine`,
results in `results/marine_toga/`). Trait: Cetacea, Pinnipedia (walrus, sea lion, monk seal) and
the manatee; 16 species, 3 origins, 29 foreground branches. Hippopotamus and giant otter
(semi-aquatic) are kept out of the 20 control trait sets.

* No gene at q = 0.1; control sets: 0 calls in all 20.
* Strongest gene: MYH13 (extraocular/superfast muscle myosin), p = 2.6e-5, q = 0.41, with
  z = 5-6.6 in all three layers. The top 20 are rich in epidermis genes (TGM3, TGM7, DSC1,
  KRTDAP, CALML3, ITGB4) and muscle glycogen/contraction genes (PYGM, PHKG1, MYOF); all are
  nominal only.
* The best-known marine gene losses (PON1, MMP12, KLK8, F12, MTNR1A, DSG4, SLC6A18) cannot be
  tested by a rate test: they have no intact 1:1 ortholog in most or all cetaceans, so those
  foreground branches are missing; MTNR1B, AANAT, ASMT, POLM, NAT16 and TAS1R1-3 failed the
  80%-of-species filter altogether. Convergent loss needs its own layer (presence/absence).
* A 105-species run with 6 aquatic origins (adds elephant seal, Weddell seal, beaver, muskrat;
  `results/toga/species_aquatic.tsv`) is in progress.

### Gene-loss layer (marine, 101 species)

`scripts/toga_loss_matrix.py` builds a gene x species status matrix from TOGA's ortholog
classification (`results/toga/loss_status_aquatic.tsv.gz`; 19,457 genes, 105 species);
`convergence_toga.py --loss` tests convergent loss (lost = L and UL; 11,815 testable genes)
with the same species-level GLS and leave-one-origin-out test.

* The raw GLS p-value is anticonservative on 0/1 data: 20 marine control sets averaged
  5.1 false loss calls and 60% had at least one. Loss p-values are therefore calibrated
  empirically against 200 matched control trait sets (stratified by number of lost
  species), and checked on 20 further control sets that the calibration never sees.

| Layer | Marine calls (q = 0.1) | Control sets: mean calls | with any call |
|---|---|---|---|
| Rate (3 layers) | 0 | 0 | 0% |
| Loss (calibrated) | 14 | 0.25 | 20% |
| Combined (rate + loss, ACAT) | 6 | 0.05 | 5% |

* Combined calls: PON1 (lost in all 15 marine species with data, in no other species),
  TAAR5, HTRA4, MYH13 (signal in all four layers), ANKRD23, OR51A7. Loss calls add
  TAS2R1, KLK8, GBA3, SULT6B1, OR51I2, MYBPHL, GLYATL3, ATP13A5.
* Known marine losses recovered: PON1, KLK8, TAAR5, TAS2R1 and olfactory receptors;
  nominal: MMP12 (p = 0.004), GSDMA, TAS1R1-3, AANAT, ASMT.

### 105-species runs (5 pinnipeds; beaver and muskrat added)

| Trait | Species / origins | Rate calls | Loss calls | Combined calls | Control sets: mean combined calls |
|---|---|---|---|---|---|
| Marine (101 species) | 16 / 3 | 0 | 14 | 6 | 0.05 |
| Marine (105 species) | 18 / 3 | 0 | 1 (ANKRD23) | 0 | 0 |
| Aquatic (105 species) | 20 / 5 | 0 | 0 | 0 | 0 |

* All runs are calibrated (rate layers: 0 false calls in every control set).
* The same genes lead in every run (PON1, KLK8, ANKRD23, MYH13, TAAR5, MYBPHL, GSDMA,
  taste and olfactory receptors), but the 105-species marine p-values are weaker: PON1
  (lost in all 16 marine species with data, in no other) has GLS p = 6e-10 but empirical
  p = 1.7e-4. Only 142 distinct matched control sets exist for this species set (200 for
  the 101-species set), and they share clades, so a few clade-specific losses recur across
  many null sets and fatten the pooled null tail. The empirical null is valid but
  conservative; a simulation-based null for gene loss is the planned fix.
* Aquatic: beaver and muskrat do not share the marine losses (PON1 and KLK8 are intact in
  both), so adding them dilutes the cross-origin signal: the convergent losses are
  specific to fully marine lineages.

### Simulation null for gene loss (marine, 105 species)

`mlconv/loss.py`: each gene is lost along the species tree at its own rate and never
regained (Dollo); each species reports intact genes as lost at its own false-loss rate
(0.2-4%, estimated from losses seen in one species only); missing data are kept as
observed. Loss rates are fitted by maximum likelihood (pruning, all genes at once). The
test statistic on 100 simulated genomes, pooled within 10 bins of fitted loss rate, is the
null; the far tail is extrapolated with a generalized Pareto fit. This replaces the
control-set null, which was limited by the number of distinct control sets.

| Loss null | Marine loss calls (q = 0.1) | Control sets: mean false calls | with any call |
|---|---|---|---|
| Control sets (142 available) | 1 | 0.15 | 10% |
| **Simulation** | **53** | **0.05** | **5%** |

* 53 convergent losses: PON1, ACSM4, ANKRD23, TGM7, GNAT3, TAS2R1, KLK8, MRGPRX2, GLYATL3,
  TAAR5, MYBPHL, CCDC170, KRT10, PTGR1, ATP13A5, OR51A7, MYH13, SULT6B1, KRT79, OR9Q2,
  OR2T1, OR5H15, OR5M9, TLE7, OR51D1, OR51B6, GBA3, OR5C1, GSDMA, OR5M3, HTRA4,
  SERPINB12, SLC4A9, KRTAP24-1, KRT1, OR5H1, OR51G1, KIAA0513, TEAD1, MRPS9, MSS51, CRYZ,
  RPL26, ANXA9, OR2K2, OR5H2, SLC5A4, OR51I2, OR51I1, TAS1R3, SLC25A45, OR5H14, RAET1E.
* Themes: chemosensation (20 olfactory, taste and trace-amine receptor genes plus the
  taste G protein GNAT3), epidermis (TGM7, KRT1, KRT10, KRT79, KRTAP24-1, SERPINB12,
  GSDMA), known marine losses (PON1, KLK8, SLC4A9).
* Several of the lowest-ranked calls (KIAA0513, TEAD1, MRPS9, CRYZ, RPL26) are lost in only
  two marine species and nowhere else, more likely annotation errors than biology; at
  q = 0.1 about five false calls are expected among 53.
* The combined rate + loss test makes no call: it covers 19,208 genes and the rate
  layer adds nothing for genes lost in the trait lineages, so the per-layer lists, each
  with its own FDR and control check, are the ones to report.

## Paper dataset: three traits, five analyses (115 species)

Species set `results/toga/species_paper.tsv` (adds mole-rats, tuco-tuco, blind mole-rat,
mole vole, wild yak, Tibetan antelope, alpaca, cane rat, honey badger); loss matrix
`results/toga/loss_status_paper.tsv.gz`; loss null = simulation with species false-loss
rate x gene fragility (`mlconv/loss.py`); 20 matched control trait sets per analysis.

| Trait | Origins | Rate calls | Loss calls | Combined calls | Results |
|---|---|---|---|---|---|
| Marine | 3 | 0 | 48 | 0 | `results/marine_toga/` |
| Hypoxia, subterranean | 6 | 0 | 15 | 10 | `results/hypoxia_subterranean_toga/` |
| Hypoxia, combined | 10 | 0 | 11 | 1 | `results/hypoxia_toga/` |
| Hypoxia, high altitude | 4 | 0 | 1 | 0 | `results/hypoxia_altitude_toga/` |
| Echolocation | 3 | 0 | 0 | 0 | `results/echolocation_toga/` |

Control sets (100 pooled): any call in 2% (rate), 8% (loss), 4% (combined).
Marine losses: chemosensation (21), epidermis (8), muscle (5: MYH13, MYH6, MYBPHL, ANKRD23,
MSS51), PON1, KLK8, SLC4A9 and others. Subterranean losses: lens and retina genes (LGSN,
LCTL, CRYGS, CRYBA4, BFSP1, GJA8, ROM1, MPP4, RGS11, CABP5, OPTC). No hypoxia or hearing
positive control was called. Earlier sections above document how each step was reached.

## Status and future work

Done for the manuscript: benchmark (Extended Data Figs. 1-3), sequence-level validation
with AliSim, real layers from Zoonomia/TOGA (relative rates, selection proxy, gene loss),
head-to-head comparison with RERconverge, application to four traits.

Not part of this manuscript (possible extensions): site-level convergent substitutions
(CSUBST, PCOC), non-coding acceleration (PhyloAcc), functional-effect scores from protein
language models, and a pip-installable package with a Zenodo DOI.

## AI-use disclosure

Nature Portfolio journals do not accept AI tools as authors and require the use of
large language models to be documented in the Methods. Code written with AI
assistance must be reviewed, tested and owned by the authors.
