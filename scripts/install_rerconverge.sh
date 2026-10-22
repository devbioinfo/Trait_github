#!/bin/bash
# Installs RERconverge (commit 682ac34, the version used in the paper) with our ggtree patch
# into the active conda environment. Usage: bash scripts/install_rerconverge.sh
set -e
cd "$(dirname "$0")/.."
REPO=$(pwd)
echo "== 1/3 supporting R packages (CRAN and Bioconductor)"
Rscript -e '
r <- "https://cloud.r-project.org"
need <- c("RColorBrewer","gplots","phytools","geiger","knitr","Rcpp","RcppArmadillo","weights",
          "phangorn","ggplot2","dplyr","FSA","castor","Matrix","TreeTools","progress","ape","BiocManager")
miss <- need[!need %in% rownames(installed.packages())]
if (length(miss)) install.packages(miss, repos = r, Ncpus = 8)
if (!"impute" %in% rownames(installed.packages())) BiocManager::install("impute", update = FALSE, ask = FALSE)
left <- c(need, "impute")[!c(need, "impute") %in% rownames(installed.packages())]
if (length(left)) { cat("NOT INSTALLED:", left, "\n"); quit(status = 1) }
cat("all supporting packages installed\n")'
echo "== 2/3 download RERconverge 682ac34 and apply the patch"
TMP=$(mktemp -d)
git clone -q https://github.com/nclark-lab/RERconverge.git "$TMP/RERconverge"
cd "$TMP/RERconverge"
git checkout -q 682ac34
git apply "$REPO/scripts/rerconverge_no_ggtree.patch"
echo "== 3/3 build and install"
R CMD INSTALL --no-test-load . > "$REPO/logs/install_rerconverge.txt" 2>&1 || { tail -30 "$REPO/logs/install_rerconverge.txt"; exit 1; }
cd "$REPO"
Rscript -e 'suppressMessages(library(RERconverge)); cat("RERconverge", as.character(packageVersion("RERconverge")), "installed OK\n")'
