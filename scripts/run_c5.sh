#!/bin/bash
# C5 locus validation of LGSN and CRYGS on the Mac. Downloads ~8 GB of genomes into data/c5/ (resumable).
#   bash scripts/run_c5.sh            (log: logs/c5_run.txt)
cd "$(dirname "$0")/.."
command -v miniprot >/dev/null || { echo "miniprot not found: install with  conda install -y -c bioconda -c conda-forge miniprot"; exit 1; }
mkdir -p logs results/c5
{ date; git rev-parse HEAD; miniprot --version; python3 --version; } > results/c5/ENVIRONMENT.txt
python3 scripts/c5_locus_validation.py --threads "${THREADS:-$(getconf _NPROCESSORS_ONLN)}" 2>&1 | tee logs/c5_run.txt
