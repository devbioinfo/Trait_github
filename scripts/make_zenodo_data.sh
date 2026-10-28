#!/bin/bash
# Builds the data deposit: zenodo/convergence_mammals_data_v1.tar.gz (+ SHA256SUMS.txt inside).
set -e
cd "$(dirname "$0")/.."
V=convergence_mammals_data_v1
T=$(mktemp -d)/$V
mkdir -p $T/{species_and_genes,branch_lengths,per_trait,supplementary,benchmarks/robustness,figures,audit,iut_check}
cp zenodo/README_data.md $T/README.md
cp results/toga/{species_paper.tsv,assemblies_paper.tsv,coverage_paper.tsv,master_paper.nwk,loss_status_paper.tsv.gz,species_aquatic.tsv,loss_status_aquatic.tsv.gz} $T/species_and_genes/
cp data/bl_paper.tar.gz $T/branch_lengths/
for t in marine hypoxia_subterranean hypoxia_altitude hypoxia echolocation aquatic; do
  mkdir -p $T/per_trait/$t; cp results/${t}_toga/* $T/per_trait/$t/
done
cp -r results/audit/* $T/audit/ && rm -rf $T/audit/flags
for t in marine hypoxia_subterranean hypoxia_altitude hypoxia; do mkdir -p $T/iut_check/$t; cp results/iut_check/$t/* $T/iut_check/$t/; done
cp results/extra/*.csv $T/supplementary/
cp results/*.csv $T/benchmarks/
cp results/robustness/*.csv $T/benchmarks/robustness/
cp -r results/signatures $T/benchmarks/
cp results/figures/*.pdf results/figures/*.png $T/figures/
(cd $T && find . -type f ! -name SHA256SUMS.txt | sort | xargs sha256sum > SHA256SUMS.txt)
tar czf zenodo/$V.tar.gz -C "$(dirname $T)" $V
ls -la zenodo/$V.tar.gz
