#!/bin/bash
# Second-audit verification of the single-origin (composite-null) result. Runs on the Mac, resumable.
#   bash scripts/run_audit2_checks.sh        (progress: logs/audit2_progress.txt)
# 1. subterranean, primary coding, 1,000,000 replicates per gene and origin (Monte Carlo floor 1e-6)
# 2-3. subterranean under the two other loss codings, 200,000 replicates, same focus genes followed
# 4. marine, primary coding, 100,000 replicates (full BH sequence)
# 5-6. simulation check of the single-origin test (0, 1, 2, 3 truly lost origins), subterranean and marine
# 7. summary -> results/audit2/summary.md
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
R=results/audit2; F=$R/flags; mkdir -p $R $F logs
[ -d data/bl_paper ] || tar xzf data/bl_paper.tar.gz -C data
W=${WORKERS:-$(getconf _NPROCESSORS_ONLN)}
{ date; git rev-parse HEAD; uname -a; python3 --version
  python3 -c "import numpy, scipy, pandas; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'pandas', pandas.__version__)"
} > $R/ENVIRONMENT.txt 2>&1
# keep the earlier 100,000-replicate per-gene file (no replicate counts) for comparison, if it is still around
for d in results/review ~/trait_local_backup/review; do
  [ -f $d/iut1e5_hypoxia_subterranean/hypoxia_subterranean_genes.csv ] && \
    cp $d/iut1e5_hypoxia_subterranean/hypoxia_subterranean_genes.csv $R/previous_iut1e5_hypoxia_subterranean_genes.csv && break
done
CT="python3 scripts/convergence_toga.py --bl data/bl_paper --species results/toga/species_paper.tsv --coverage results/toga/coverage_paper.tsv --loss results/toga/loss_status_paper.tsv.gz --workers $W --controls 0"
FOCUS=CRYGS,LGSN,MPP4,CABP5,LCTL,BFSP1,ROM1,RGS11,LACTBL1,RSKR,TPD52L3
L=$(mktemp)
cat > $L <<CMDS
iut_primary_subterranean|$CT --trait hypoxia_subterranean --iut-sims 1000000 --iut-genes $FOCUS --out $R/iut_primary_subterranean
iut_Lonly_subterranean|$CT --trait hypoxia_subterranean --loss-states L --iut-sims 200000 --iut-genes $FOCUS --out $R/iut_Lonly_subterranean
iut_ULintact_subterranean|$CT --trait hypoxia_subterranean --loss-states L --intact-states I,PI,UL --iut-sims 200000 --iut-genes $FOCUS --out $R/iut_ULintact_subterranean
iut_primary_marine|$CT --trait marine --iut-sims 100000 --out $R/iut_primary_marine
sim_subterranean|python3 scripts/sim_composite_null.py --trait hypoxia_subterranean --genes 300 --sims 2000 --workers $W --out $R
sim_marine|python3 scripts/sim_composite_null.py --trait marine --genes 300 --sims 2000 --workers $W --out $R
summary|python3 scripts/audit2_summary.py
CMDS
cp $L $R/commands.txt
run() { name=${1%%|*}; cmd=${1#*|}; [ -f $F/$name.done ] && [ "$name" != summary ] && return 0
  echo "$(date '+%d %b %H:%M')  started  $name" >> logs/audit2_progress.txt
  if $cmd < /dev/null > logs/audit2_$name.txt 2>&1; then touch $F/$name.done; echo "$(date '+%d %b %H:%M')  finished $name" >> logs/audit2_progress.txt
  else echo "$(date '+%d %b %H:%M')  FAILED   $name (see logs/audit2_$name.txt)" >> logs/audit2_progress.txt; fi; }
n=$(wc -l < $L); i=0
while IFS= read -r line; do i=$((i+1)); echo "step $i of $n: ${line%%|*}"; run "$line"; done < $L
echo "All done: $(ls $F | wc -l | tr -d ' ') of $((n-1)) steps finished (summary always reruns)."
