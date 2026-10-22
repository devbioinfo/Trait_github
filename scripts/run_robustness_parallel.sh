#!/bin/bash
# Parallel version of run_robustness.sh: same steps and flags, 3 at a time.
cd /home/user/trait_github
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
R=results/robustness; F=/home/user/data_toga/robust_flags; mkdir -p $R $F
FB="python3 -I scripts/final_benchmark.py --reps 1"
JOBS=$(mktemp)
for i in 4 5 6 7 8 9; do echo "main_$i|$FB --seed $((3000+i)) --tree-label $i --out $R/main_tree$i.csv"; done >> $JOBS
for i in 0 1 2; do
  echo "tips100_$i|$FB --tips 100 --confounding 1 --seed $((4100+i)) --tree-label $i --out $R/sens_tips100_$i.csv"
  echo "frac005_$i|$FB --frac 0.005 --confounding 1 --seed $((4200+i)) --tree-label $i --out $R/sens_frac0.005_$i.csv"
  echo "frac05_$i|$FB --frac 0.05 --confounding 1 --seed $((4300+i)) --tree-label $i --out $R/sens_frac0.05_$i.csv"
  echo "frac10_$i|$FB --frac 0.10 --confounding 1 --seed $((4400+i)) --tree-label $i --out $R/sens_frac0.10_$i.csv"
  echo "disc10_$i|$FB --discordance 10 --confounding 1 --seed $((4500+i)) --tree-label $i --out $R/sens_disc10_$i.csv"
  echo "disc40_$i|$FB --discordance 40 --confounding 1 --seed $((4600+i)) --tree-label $i --out $R/sens_disc40_$i.csv"
  echo "tips400_$i|$FB --tips 400 --confounding 1 --seed $((4700+i)) --tree-label $i --out $R/sens_tips400_$i.csv"
done >> $JOBS
echo "rer_allperm|python3 -I scripts/rerconverge_compare.py --confounding 1 --effects moderate --perms 100 --candidates 0 --seed 5031 --out $R/rerconverge_allperm.csv" >> $JOBS
echo "origins|python3 -I scripts/origins_power.py --reps 7 --seed 5011 --rep-offset 3 --out $R/origins_power_extra.csv" >> $JOBS
echo "loss|python3 -I scripts/loss_benchmark.py --bl /home/user/data_toga/bl_paper --loss results/toga/loss_status_paper.tsv.gz --out results/loss_benchmark.csv --reps 5" >> $JOBS
run() { name=${1%%|*}; cmd=${1#*|}; [ -f $F/$name.done ] && exit 0
  echo "== $name start $(date '+%d %H:%M')"; $cmd > /home/user/data_toga/logs_$name.txt 2>&1 && touch $F/$name.done && echo "== $name done $(date '+%d %H:%M')"; }
export -f run; export F
tr '\n' '\0' < $JOBS | xargs -0 -P 3 -I{} bash -c 'run "$@"' _ {}
touch $F/all.done
