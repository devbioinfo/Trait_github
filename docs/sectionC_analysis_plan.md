# Section C: does recurrent gene loss track visual regression or subterranean habitat?

Pre-specified analysis plan, fixed before any of the analyses below is run. Changes after this
commit are recorded in the git history with their reason.

## Question
The subterranean candidates are dominated by lens and retinal genes. Is recurrent loss better explained
by **visual regression** (reduced or vestigial eyes) than by **subterranean habitat**, and does a set of
losses discovered in some origins **predict** losses in origins withheld from discovery?

## Phenotypes (`data/phenotypes/visual_phenotypes.csv`)
- `eye_state`: 0 functional eyes; 1 reduced eyes with residual light detection or vision; 2 vestigial,
  subcutaneous or skin-covered eyes without image-forming vision.
- `fossoriality`: 0 surface; 1 burrowing; 2 subterranean (lives and forages mainly underground).
- Species not listed are coded 0/0. Every coding is checked against the literature by the authors
  (column `source`) before the analyses are run; uncertainty is recorded and used in sensitivity analyses.

Derived traits:
| Trait | Definition | Origins in the 115-species tree |
|---|---|---|
| S (habitat, as in the main analysis) | fossoriality = 2 | 6 |
| V (visual regression, primary) | eye_state >= 1, excluding pangolins | 6: African mole-rats, blind mole-rat, mole vole, moles, golden mole, river dolphins |
| V+ (sensitivity) | eye_state >= 1, including pangolins | 7 |
| V2 (strict) | eye_state = 2 | 3: blind mole-rat, eastern mole, golden mole |

Discordant lineages carry the information: the tuco-tuco (S, not V) and the river dolphins (V, not S);
pangolins in V+ only. With so few discordant origins, power is limited and this is stated in advance.

## Analyses
1. **Trait runs.** V, V+ and V2 through the complete pipeline (L+UL coding, simulated loss null,
   single-origin check, conservative composite-null BH), each with 100 matched control traits.
   Compare call sets with S (shared genes, Jaccard index, functional groups).
2. **Joint model.** For every loss-tested gene, GLS with both indicators (V and S) and the
   genome-learned species covariance; partial effects as in Fig. 4c. Pre-specified gene sets: the 15 S
   candidates, the V candidates, and the ocular genes of Fig. 6a. Test: is the mean partial effect of V
   larger than that of S within each set, against a null of loss-frequency-matched gene sets (1,000 draws)?
3. **Withheld-origin prediction (frozen rules).**
   - Discovery rule: global loss test, L+UL coding, simulated null with 100 replicates, BH q = 0.1;
     species of the withheld origin set to missing, loss model refitted.
   - (a) Within S: each of the 6 origins withheld in turn. (b) Out of habitat: discovery on S with all
     6 origins, prediction in the river dolphins (V, not S) and, separately, in the pangolins.
   - Evaluation statistic: the share of predicted genes lost in at least one species of the withheld
     lineage. Null: 1,000 sets of genes matched to the predicted genes on total losses across the
     115 species and on missing data in the withheld lineage. Calibration: the same statistic for
     100 control lineages matched to the withheld lineage in species number and annotated loss burden.
   - Every fold is reported, including failures.
4. **Convergence across levels.** For the S and V origins, the number of origins with a loss at the level
   of (i) the exact gene, (ii) the module (lens; photoreceptor and phototransduction; ON-bipolar
   signalling; gene lists of Fig. 6a fixed now) and (iii) the function (GO:0002088 lens development in
   camera-type eye; GO:0007601 visual perception). Null: genes replaced by loss-frequency-matched genes,
   one representative per gene family (as in the family-collapsed enrichment).
5. **Independent validation of priority loci** (LGSN, CRYGS): lesions checked against independent
   assemblies or sequencing reads where available; synteny and coverage of the locus.

## Interpretation fixed in advance
- Support for visual regression over habitat requires the V partial effect to exceed the S partial
  effect for the ocular set **and** the subterranean ocular losses to predict losses in the river
  dolphins above the matched-gene null and the control-lineage calibration.
- A null result in the river dolphins is reported as such; with one non-subterranean origin it cannot
  exclude visual regression as the driver.
- Associations do not establish adaptive loss or mechanism.
