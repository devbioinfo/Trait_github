#!/bin/bash
# Reruns the main analyses with the per-origin (intersection-union) calibrated gene-loss test.
# Usage (from the repository folder):
#   nohup caffeinate -i bash scripts/run_iut.sh > iut.out 2>&1 &
# Resumable: finished steps are skipped. Progress: logs/iut_progress.txt
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
A=results/iut; F=$A/flags; mkdir -p $A $F logs
[ -d data/bl_paper ] || tar xzf data/bl_paper.tar.gz -C data
W=${WORKERS:-$(getconf _NPROCESSORS_ONLN)}
CT="python3 scripts/convergence_toga.py --bl data/bl_paper --species results/toga/species_paper.tsv --coverage results/toga/coverage_paper.tsv --loss results/toga/loss_status_paper.tsv.gz --workers $W --loss-test iut"
L=$(mktemp)
TRAITS="marine hypoxia_subterranean hypoxia_altitude hypoxia echolocation"
# 1. main analyses, 100 matched control traits each
for t in $TRAITS; do echo "main_$t|$CT --trait $t --controls 100 --out $A/$t"; done >> $L
# 2. TOGA 'lost' only (uncertain losses missing), also with 100 control traits
for t in $TRAITS; do echo "Lonly_$t|$CT --trait $t --loss-states L --controls 100 --out $A/Lonly_$t"; done >> $L
# 3. marine control traits matched also on annotated losses per species
echo "lm_marine|$CT --trait marine --controls 100 --match-losses --out $A/lm_marine" >> $L
# 4. null resolution: 1,000 simulated genomes
for t in marine hypoxia_subterranean; do echo "sims1000_$t|$CT --trait $t --loss-sims 1000 --controls 0 --out $A/sims1000_$t"; done >> $L
run() { name=${1%%|*}; cmd=${1#*|}; [ -f $F/$name.done ] && return 0
  echo "$(date '+%d %b %H:%M')  started  $name" >> logs/iut_progress.txt
  if $cmd < /dev/null > logs/iut_$name.txt 2>&1; then touch $F/$name.done; echo "$(date '+%d %b %H:%M')  finished $name" >> logs/iut_progress.txt
  else echo "$(date '+%d %b %H:%M')  FAILED   $name (see logs/iut_$name.txt)" >> logs/iut_progress.txt; fi; }
# a plain loop: macOS xargs -I rejects command lines longer than 255 characters
while IFS= read -r line; do [ -n "$line" ] && run "$line"; done < $L
n=$(wc -l < $L); d=$(ls $F | wc -l)
echo "All done: $d of $n steps finished." | tee -a logs/iut_progress.txt
