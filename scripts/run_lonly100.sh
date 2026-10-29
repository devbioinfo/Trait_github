#!/bin/bash
# Confident-loss coding (TOGA 'L' only; uncertain losses missing) with 100 matched control traits.
# Usage (from the repository folder):
#   nohup caffeinate -i bash scripts/run_lonly100.sh > lonly100.out 2>&1 &
# Resumable: finished steps are skipped. Progress: logs/lonly100_progress.txt
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
A=results/audit; F=$A/flags; mkdir -p $A $F logs
[ -d data/bl_paper ] || tar xzf data/bl_paper.tar.gz -C data
W=${WORKERS:-$(getconf _NPROCESSORS_ONLN)}
CT="python3 scripts/convergence_toga.py --bl data/bl_paper --species results/toga/species_paper.tsv --coverage results/toga/coverage_paper.tsv --loss results/toga/loss_status_paper.tsv.gz --workers $W"
L=$(mktemp)
for t in marine hypoxia_subterranean hypoxia_altitude hypoxia echolocation; do
  echo "Lonly100_$t|$CT --trait $t --loss-states L --controls 100 --out $A/Lonly100_$t"
done >> $L
run() { name=${1%%|*}; cmd=${1#*|}; [ -f $F/$name.done ] && return 0
  echo "$(date '+%d %b %H:%M')  started  $name" >> logs/lonly100_progress.txt
  if $cmd < /dev/null > logs/lonly100_$name.txt 2>&1; then touch $F/$name.done; echo "$(date '+%d %b %H:%M')  finished $name" >> logs/lonly100_progress.txt
  else echo "$(date '+%d %b %H:%M')  FAILED   $name (see logs/lonly100_$name.txt)" >> logs/lonly100_progress.txt; fi; }
# a plain loop: macOS xargs -I rejects command lines longer than 255 characters
while IFS= read -r line; do [ -n "$line" ] && run "$line"; done < $L
n=$(wc -l < $L); d=$(ls $F | grep -c Lonly100)
echo "All done: $d of $n steps finished." | tee -a logs/lonly100_progress.txt
