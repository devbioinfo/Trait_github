# Export RERconverge's bundled mammal protein gene trees (mamTreesSub: 1,000 genes, 62 species)
# for genes present in all species: the master tree and one line per gene, "<gene>\t<newick>".
# Usage: Rscript export_rerconverge_trees.R <RERconverge source dir> <out dir>
suppressMessages(library(ape))
args <- commandArgs(trailingOnly = TRUE)
obj <- load(file.path(args[1], "data", "mamTreesSub.RData"))
x <- get(obj[1])
master <- x$masterTree
write.tree(master, file.path(args[2], "master.nwk"))
n <- length(master$tip.label)
keep <- names(x$trees)[sapply(x$trees, function(t) length(t$tip.label) == n)]
lines <- vapply(keep, function(g) paste0(g, "\t", write.tree(x$trees[[g]])), "")
writeLines(lines, file.path(args[2], "gene_trees.tsv"))
cat(length(keep), "complete genes of", length(x$trees), "\n")
