#!/bin/bash
# Robustness queue for the manuscript. Resumable: each step leaves a .done flag.
cd /home/user/trait_github
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
R=results/robustness; F=/home/user/data_toga/robust_flags; mkdir -p $R $F
step() { local name=$1; shift; [ -f $F/$name.done ] && return; echo "== $name $(date +%H:%M)"; "$@" && touch $F/$name.done; }
FB="python3 -I scripts/final_benchmark.py --reps 1"
# 1. main benchmark: 7 more replicate trees (10 in total)
for i in 3 4 5 6 7 8 9; do step main_$i $FB --seed $((3000+i)) --tree-label $i --out $R/main_tree$i.csv; done
# 2. sensitivity, confounding 1, three trees each
for i in 0 1 2; do
  step tips100_$i $FB --tips 100 --confounding 1 --seed $((4100+i)) --tree-label $i --out $R/sens_tips100_$i.csv
  step frac005_$i $FB --frac 0.005 --confounding 1 --seed $((4200+i)) --tree-label $i --out $R/sens_frac0.005_$i.csv
  step frac05_$i $FB --frac 0.05 --confounding 1 --seed $((4300+i)) --tree-label $i --out $R/sens_frac0.05_$i.csv
  step frac10_$i $FB --frac 0.10 --confounding 1 --seed $((4400+i)) --tree-label $i --out $R/sens_frac0.10_$i.csv
  step disc10_$i $FB --discordance 10 --confounding 1 --seed $((4500+i)) --tree-label $i --out $R/sens_disc10_$i.csv
  step disc40_$i $FB --discordance 40 --confounding 1 --seed $((4600+i)) --tree-label $i --out $R/sens_disc40_$i.csv
  step tips400_$i $FB --tips 400 --confounding 1 --seed $((4700+i)) --tree-label $i --out $R/sens_tips400_$i.csv
done
# 3. RERconverge with permulations on all genes (one tree, confounding 1, moderate effect)
step rer_allperm python3 -I scripts/rerconverge_compare.py --confounding 1 --effects moderate --perms 100 --candidates 0 --seed 5031 --out $R/rerconverge_allperm.csv
# 4. origins benchmark: 7 more replicates
step origins python3 -I scripts/origins_power.py --reps 7 --seed 5011 --rep-offset 3 --out $R/origins_power_extra.csv
# 5. loss benchmark: replicates 2-4
step loss python3 -I scripts/loss_benchmark.py --bl /home/user/data_toga/bl_paper --loss results/toga/loss_status_paper.tsv.gz --out results/loss_benchmark.csv --reps 5
touch $F/all.done
