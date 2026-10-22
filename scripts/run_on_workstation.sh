#!/bin/bash
# Run every remaining robustness analysis on your own computer.
# Usage (from the repository folder):  bash scripts/run_on_workstation.sh
# Safe to stop and start again: finished steps are skipped.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
R=results/robustness; F=results/robustness/flags; mkdir -p $R $F logs
[ -d data/bl_paper ] || tar xzf data/bl_paper.tar.gz -C data
W=${WORKERS:-$(nproc)}
echo "Using $W cores. Progress is written to logs/progress.txt"
FB="python3 scripts/final_benchmark.py --reps 1 --workers $W"
L=$(mktemp)
for i in 3 4 5 6 7 8 9; do echo "main_$i|$FB --seed $((3000+i)) --tree-label $i --out $R/main_tree$i.csv"; done >> $L
for i in 0 1 2; do
  echo "tips100_$i|$FB --tips 100 --confounding 1 --seed $((4100+i)) --tree-label $i --out $R/sens_tips100_$i.csv"
  echo "frac005_$i|$FB --frac 0.005 --confounding 1 --seed $((4200+i)) --tree-label $i --out $R/sens_frac0.005_$i.csv"
  echo "frac05_$i|$FB --frac 0.05 --confounding 1 --seed $((4300+i)) --tree-label $i --out $R/sens_frac0.05_$i.csv"
  echo "frac10_$i|$FB --frac 0.10 --confounding 1 --seed $((4400+i)) --tree-label $i --out $R/sens_frac0.10_$i.csv"
  echo "disc10_$i|$FB --discordance 10 --confounding 1 --seed $((4500+i)) --tree-label $i --out $R/sens_disc10_$i.csv"
  echo "disc40_$i|$FB --discordance 40 --confounding 1 --seed $((4600+i)) --tree-label $i --out $R/sens_disc40_$i.csv"
  echo "tips400_$i|$FB --tips 400 --confounding 1 --seed $((4700+i)) --tree-label $i --out $R/sens_tips400_$i.csv"
done >> $L
echo "rer_allperm|python3 scripts/rerconverge_compare.py --confounding 1 --effects moderate --perms 100 --candidates 0 --seed 5031 --out $R/rerconverge_allperm.csv" >> $L
echo "origins|python3 scripts/origins_power.py --workers $W --reps 7 --seed 5011 --rep-offset 3 --out $R/origins_power_extra.csv" >> $L
echo "loss|python3 scripts/loss_benchmark.py --workers $W --bl data/bl_paper --loss results/toga/loss_status_paper.tsv.gz --out $R/loss_benchmark_5reps.csv --reps 5" >> $L
run() { name=${1%%|*}; cmd=${1#*|}; [ -f $F/$name.done ] && exit 0
  echo "$(date '+%d %b %H:%M')  started  $name" >> logs/progress.txt
  if $cmd > logs/$name.txt 2>&1; then touch $F/$name.done; echo "$(date '+%d %b %H:%M')  finished $name" >> logs/progress.txt
  else echo "$(date '+%d %b %H:%M')  FAILED   $name (see logs/$name.txt)" >> logs/progress.txt; fi; }
export -f run; export F
tr '\n' '\0' < $L | xargs -0 -P 1 -I{} bash -c 'run "$@"' _ {}
echo "All done: $(ls $F | wc -l) of 26 steps finished." | tee -a logs/progress.txt
