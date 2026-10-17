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
