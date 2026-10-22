#!/bin/bash
# Checks that this computer has what the robustness runs need.
ok=1
command -v python3 >/dev/null && echo "OK   python3" || { echo "MISSING python3"; ok=0; }
python3 -c "import numpy, scipy, pandas" 2>/dev/null && echo "OK   numpy/scipy/pandas" || { echo "MISSING python packages: pip install numpy scipy pandas"; ok=0; }
command -v iqtree2 >/dev/null && echo "OK   iqtree2" || { echo "MISSING iqtree2 (conda install -c bioconda iqtree)"; ok=0; }
command -v Rscript >/dev/null && Rscript -e 'library(RERconverge)' >/dev/null 2>&1 && echo "OK   R + RERconverge" || { echo "MISSING R/RERconverge (needed only for one step)"; }
[ $ok = 1 ] && echo "Ready. Run: bash scripts/run_on_workstation.sh" || echo "Install the missing items above, then run this check again."
