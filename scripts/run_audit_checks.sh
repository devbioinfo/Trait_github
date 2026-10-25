#!/bin/bash
# Checks requested by the pre-submission audit. Usage (from the repository folder):
#   nohup caffeinate -i bash scripts/run_audit_checks.sh > audit.out 2>&1 &
# Resumable: finished steps are skipped. Progress: logs/audit_progress.txt
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
A=results/audit; F=$A/flags; mkdir -p $A $F logs
[ -d data/bl_paper ] || tar xzf data/bl_paper.tar.gz -C data
W=${WORKERS:-$(getconf _NPROCESSORS_ONLN)}
CT="python3 scripts/convergence_toga.py --bl data/bl_paper --species results/toga/species_paper.tsv --coverage results/toga/coverage_paper.tsv --loss results/toga/loss_status_paper.tsv.gz --workers $W"
FB="python3 scripts/final_benchmark.py --reps 1 --workers $W"
L=$(mktemp)
TRAITS="marine hypoxia_subterranean hypoxia_altitude hypoxia echolocation"
# 1. 100 matched control traits per analysis (calibration with binomial intervals, null P-value plots)
for t in $TRAITS; do echo "ctrl100_$t|$CT --trait $t --controls 100 --out $A/ctrl100_$t"; done >> $L
# 2. gene loss counted as TOGA 'lost' only (uncertain losses treated as intact)
for t in $TRAITS; do echo "Lonly_$t|$CT --trait $t --loss-states L --controls 20 --out $A/Lonly_$t"; done >> $L
# 3. loss-null resolution: 500 simulated genomes, and the tail fit on 100 or 500 null values
for t in marine hypoxia_subterranean; do
  echo "sims500_$t|$CT --trait $t --loss-sims 500 --controls 0 --out $A/sims500_$t"
  echo "tail100_$t|$CT --trait $t --tail-n 100 --controls 0 --out $A/tail100_$t"
  echo "tail500_$t|$CT --trait $t --tail-n 500 --controls 0 --out $A/tail500_$t"
done >> $L
# 4. two more trees for every sensitivity setting (5 per setting)
R=results/robustness
for i in 3 4; do
  echo "tips100_$i|$FB --tips 100 --confounding 1 --seed $((4100+i)) --tree-label $i --out $R/sens_tips100_$i.csv"
  echo "frac005_$i|$FB --frac 0.005 --confounding 1 --seed $((4200+i)) --tree-label $i --out $R/sens_frac0.005_$i.csv"
  echo "frac05_$i|$FB --frac 0.05 --confounding 1 --seed $((4300+i)) --tree-label $i --out $R/sens_frac0.05_$i.csv"
  echo "frac10_$i|$FB --frac 0.10 --confounding 1 --seed $((4400+i)) --tree-label $i --out $R/sens_frac0.10_$i.csv"
  echo "disc10_$i|$FB --discordance 10 --confounding 1 --seed $((4500+i)) --tree-label $i --out $R/sens_disc10_$i.csv"
  echo "disc40_$i|$FB --discordance 40 --confounding 1 --seed $((4600+i)) --tree-label $i --out $R/sens_disc40_$i.csv"
  echo "tips400_$i|$FB --tips 400 --confounding 1 --seed $((4700+i)) --tree-label $i --out $R/sens_tips400_$i.csv"
done >> $L
# 5. RERconverge with 10,000 permulations on all genes at rate-jump intensity 0 and 2
echo "rer10k_conf02|env RER_CORES=$(( W > 4 ? W - 4 : 1 )) python3 scripts/rerconverge_compare.py --confounding 0,2 --effects moderate --perms 10000 --candidates 0 --seed 5032 --workers $W --out $R/rerconverge_allperm10k_conf02.csv" >> $L
run() { name=${1%%|*}; cmd=${1#*|}; [ -f $F/$name.done ] && exit 0
  echo "$(date '+%d %b %H:%M')  started  $name" >> logs/audit_progress.txt
  if $cmd > logs/audit_$name.txt 2>&1; then touch $F/$name.done; echo "$(date '+%d %b %H:%M')  finished $name" >> logs/audit_progress.txt
  else echo "$(date '+%d %b %H:%M')  FAILED   $name (see logs/audit_$name.txt)" >> logs/audit_progress.txt; fi; }
export -f run; export F
tr '\n' '\0' < $L | xargs -0 -P 1 -I{} bash -c 'run "$@"' _ {}
n=$(wc -l < $L); d=$(ls $F | wc -l)
echo "All done: $d of $n steps finished." | tee -a logs/audit_progress.txt
