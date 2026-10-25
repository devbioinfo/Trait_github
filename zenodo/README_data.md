# Data for "Convergent evolution in mammals is written in gene loss"

Derived data behind every result and figure of the article. Code: [DOI of the software
deposit] (GitHub: github.com/devbioinfo/Trait_github). Licence: CC BY 4.0.

Inputs are public and are not redistributed here: TOGA orthology, loss classifications,
inactivating mutations and codon alignments for the human hg38 reference
(genome.senckenberg.de/download/TOGA; Kirilenko et al. 2023, Science 380, eabn3107); the
Zoonomia 447-species tree (Zoonomia Consortium 2020); Human Protein Atlas consensus tissue
expression; Reactome pathways; Gene Ontology (go-basic.obo, goa_human.gaf).

`SHA256SUMS.txt` lists a checksum for every file.

## species_and_genes/
| File | Contents |
|---|---|
| `species_paper.tsv` | The 115 species and the TOGA assembly used for each |
| `assemblies_paper.tsv`, `coverage_paper.tsv` | Assembly metadata, common names, taxonomic lineage and 1:1-ortholog coverage |
| `master_paper.nwk` | Species tree (Zoonomia topology pruned to the 115 species) |
| `loss_status_paper.tsv.gz` | Gene x species TOGA status: I intact, PI partially intact, UL uncertain loss, L lost, M missing |
| `species_aquatic.tsv`, `loss_status_aquatic.tsv.gz` | Same for the 105-species set used while developing the loss null |

## branch_lengths/
`bl_paper.tar.gz`: per-gene branch lengths fitted with IQ-TREE 2 on the fixed topology
(`branch_lengths.jsonl`, one gene per line: GTR+G4 codon-alignment and LG+G4 protein
branch lengths per master-tree branch; `master.nwk`). Input to the three rate layers.

## per_trait/
One folder per analysis (`marine`, `hypoxia_subterranean`, `hypoxia_altitude`,
`hypoxia` = combined, `echolocation`, `aquatic`):
| File | Contents |
|---|---|
| `<trait>_genes.csv` | Every tested gene: rate-layer z-scores and P, loss P (simulated null and GLS), trait species with the gene lost, combined P, calls at q = 0.1, convergence signature |
| `control_calls.csv` | The 20 matched control traits: species and number of calls per layer |
| `summary.json` | Counts of genes, species, origins, calls and control-trait calibration |
| `trait_species.txt` | Species coded as having the trait |
| `signature_patterns.csv` | Fitted weights of the convergence-signature patterns |

## supplementary/
`supplementary_table1_convergent_losses.csv` (Supplementary Table 1, the 63 convergent
losses); enrichment results (`enrichment_tissue.csv`, `enrichment_reactome.csv`,
`enrichment_go.csv`); mutations and exons behind the marine losses
(`marine_detail_mutations.csv`, `marine_detail_exons.csv`, `marine_origin_funnel.csv`);
codon alignments of PON1 and LGSN (`codon_alignment_detail.csv`); timing of losses
(`loss_timing.csv`); correlated-trait models (`joint_marine_semiaquatic.csv`,
`joint_subterranean_altitude.csv`); origin subsampling (`origin_subsampling.csv`);
loss-model parameters and adequacy checks (`loss_model_parameters.csv`,
`adequacy_losses_per_gene.csv`, `adequacy_losses_per_species.csv`,
`species_quality.csv`).

## benchmarks/
Simulation results (one row per method x setting): `final_benchmark.csv` and
`robustness/main_tree3-9.csv` (main benchmark, 10 trees; `main_tree3_mac.csv` is tree 3
rerun on macOS ARM), `robustness/sens_*.csv` (sensitivity), `origins_power.csv` and
`robustness/origins_power_extra.csv` (number of origins), `loss_benchmark.csv`
(gene-loss layer), `rerconverge_compare.csv` and `robustness/rerconverge_allperm*.csv`
(RERconverge), `alisim_realistic.csv`, `signatures/` (signature evaluation),
and earlier development benchmarks (`benchmark_v0-v2.csv`).

## figures/
All main and Extended Data figures and the study summary as PDF (editable text) and PNG.
