#!/bin/bash
# Analyses requested by the external review (31 Oct 2026). Runs locally; resumable; logs everything.
# Usage (from the repository folder, conda env active):
#   nohup caffeinate -i bash scripts/run_review_checks.sh > review.out 2>&1 &
# Progress: logs/review_progress.txt   Outputs: results/review/   Summary: results/review/summary.md
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
R=results/review; F=$R/flags; mkdir -p $R $F logs data/annot
[ -d data/bl_paper ] || tar xzf data/bl_paper.tar.gz -C data
W=${WORKERS:-$(getconf _NPROCESSORS_ONLN)}
# environment record
{ date; git rev-parse HEAD; uname -a; python3 --version
  python3 -c "import numpy, scipy, pandas; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'pandas', pandas.__version__)"
  conda list 2>/dev/null | head -200; } > $R/ENVIRONMENT.txt 2>&1
CT="python3 scripts/convergence_toga.py --bl data/bl_paper --species results/toga/species_paper.tsv --coverage results/toga/coverage_paper.tsv --loss results/toga/loss_status_paper.tsv.gz --workers $W"
L=$(mktemp)
for t in marine hypoxia_subterranean; do
  # 1. finer per-gene single-origin check (100,000 replicates per gene and origin)
  echo "iut1e5_$t|$CT --trait $t --controls 0 --iut-sims 100000 --out $R/iut1e5_$t"
  # 2. covariance shrinkage towards the identity
  for a in 0.1 0.3; do echo "shrink${a}_$t|env MLCONV_COV_SHRINK=$a $CT --trait $t --controls 0 --out $R/shrink${a}_$t"; done
  # 3. loss-model alternatives: false-intact probability and false-loss cap
  echo "falseintact0.01_$t|env MLCONV_FALSE_INTACT=0.01 $CT --trait $t --controls 0 --out $R/falseintact0.01_$t"
  for c in 0.2 0.9; do echo "cap${c}_$t|env MLCONV_LOSS_CAP=$c $CT --trait $t --controls 0 --out $R/cap${c}_$t"; done
done >> $L
# 4. third coding: uncertain losses counted as intact (common-mask sensitivity), 20 control traits each
for t in marine hypoxia_subterranean hypoxia_altitude hypoxia echolocation; do
  echo "ULintact_$t|$CT --trait $t --loss-states L --intact-states I,PI,UL --controls 20 --out $R/ULintact_$t"
done >> $L
# 5. enrichment with and without collapsing gene families (annotation versions recorded)
echo "annot|bash scripts/get_annotations.sh data/annot" >> $L
echo "enrich|python3 scripts/function_enrichment.py --annot data/annot --suffix _review" >> $L
echo "enrich_collapsed|python3 scripts/function_enrichment.py --annot data/annot --collapse-families --suffix _review_collapsed" >> $L
# 6. summary (held-out controls, composite-null tier, call-set stability, enrichment comparison)
echo "summary|python3 scripts/review_checks_summary.py" >> $L
cp $L $R/commands.txt
run() { name=${1%%|*}; cmd=${1#*|}; [ -f $F/$name.done ] && [ "$name" != summary ] && return 0
  echo "$(date '+%d %b %H:%M')  started  $name" >> logs/review_progress.txt
  if $cmd < /dev/null > logs/review_$name.txt 2>&1; then touch $F/$name.done; echo "$(date '+%d %b %H:%M')  finished $name" >> logs/review_progress.txt
  else echo "$(date '+%d %b %H:%M')  FAILED   $name (see logs/review_$name.txt)" >> logs/review_progress.txt; fi; }
# a plain loop: macOS xargs -I rejects command lines longer than 255 characters
while IFS= read -r line; do [ -n "$line" ] && run "$line"; done < $L
n=$(wc -l < $L); d=$(ls $F | wc -l)
echo "All done: $d of $n steps finished." | tee -a logs/review_progress.txt
