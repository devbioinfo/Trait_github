# Parametric RERconverge for many binary foregrounds on one set of gene trees.
# Usage: Rscript rer_foregrounds.R <trees.tsv> <foregrounds.txt> <out_rho.csv> <out_p.csv>
# foregrounds.txt: one foreground per line, species separated by commas.
# Relative rates are computed once; each foreground is tested with
# foreground2Tree(clade = "all") + correlateWithBinaryPhenotype(weighted = "auto").
suppressMessages(library(RERconverge))
args <- commandArgs(trailingOnly = TRUE)
trees <- readTrees(args[1], minTreesAll = 20)
rer <- getAllResiduals(trees, transform = "sqrt", min.sp = 10, norm = "scale")
fgs <- strsplit(readLines(args[2]), ",")
rho <- matrix(NA_real_, nrow(rer), length(fgs), dimnames = list(rownames(rer), NULL))
p <- rho
for (i in seq_along(fgs)) {
  fg_tree <- foreground2Tree(fgs[[i]], trees, clade = "all", plotTree = FALSE)
  phen <- tree2Paths(fg_tree, trees)
  cors <- suppressWarnings(correlateWithBinaryPhenotype(rer, phen, min.sp = 10, min.pos = 2,
                                                         weighted = "auto"))
  rho[rownames(cors), i] <- cors$Rho
  p[rownames(cors), i] <- cors$P
}
write.csv(rho, args[3])
write.csv(p, args[4])
