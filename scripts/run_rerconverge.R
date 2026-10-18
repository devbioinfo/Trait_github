# Run RERconverge (binary trait) on one simulated genome.
# Usage: Rscript run_rerconverge.R <trees.tsv> <foreground_tips.txt> <root_species> <n_perms> <out.csv> [n_candidates]
# With n_candidates > 0, permulations are run only for the n_candidates genes with the smallest
# parametric P (the usual filter-then-permulate practice); other genes get perm_p = NA.
# trees.tsv: one gene per line, "<gene>\t<newick>", all on the same topology.
suppressMessages(library(RERconverge))
args <- commandArgs(trailingOnly = TRUE)
tree_file <- args[1]; fg_file <- args[2]; root_sp <- args[3]
n_perms <- as.integer(args[4]); out_file <- args[5]
n_cand <- if (length(args) >= 6) as.integer(args[6]) else 0

trees <- readTrees(tree_file, minTreesAll = 20)
fg <- readLines(fg_file)
rer <- getAllResiduals(trees, transform = "sqrt", min.sp = 10, norm = "scale")
fg_tree <- foreground2Tree(fg, trees, clade = "all", plotTree = FALSE)
phen <- tree2Paths(fg_tree, trees)
cors <- correlateWithBinaryPhenotype(rer, phen, min.sp = 10, min.pos = 2, weighted = "auto")
out <- data.frame(gene = rownames(cors), rho = cors$Rho, p = cors$P, p_adj = cors$p.adj)

if (n_perms > 0) {
  inp <- getBinaryPermulationInputsFromTree(fg_tree)
  rer_p <- rer
  cors_p <- cors
  if (n_cand > 0) {
    cand <- head(rownames(cors)[order(cors$P)], n_cand)
    rer_p <- rer[cand, , drop = FALSE]
    cors_p <- cors[cand, ]
  }
  perms <- getPermsBinary(n_perms, inp$fg_vec, inp$sisters_list, root_sp, rer_p, trees,
                          trees$masterTree, permmode = "cc")
  pp <- permpvalcor(cors_p, perms)
  out$perm_p <- pp[out$gene, "permpval"]      # (num + 1) / (denom + 1) on the side of the median
  out$perm_stat <- pp[out$gene, "permstats"]  # -log10(perm_p) * sign(Rho - median permuted Rho)
}
write.csv(out, out_file, row.names = FALSE)
