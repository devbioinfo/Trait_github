"""Gene coverage of each TOGA query assembly across the multiple codon alignments.

Reads only the FASTA headers of every *.fasta.gz in the alignment directory and reports,
for each assembly that is also in the Zoonomia 447-way tree, the fraction of genes in
which it has exactly one sequence (1:1 ortholog), together with its species name and
lineage from TOGA's overview table.

Usage: python -I scripts/toga_species_coverage.py <mca_dir> <overview.table.tsv> <447way.nh> <out.tsv>
"""
from __future__ import annotations

import csv
import gzip
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlconv.tree import from_newick  # noqa: E402


def main(mca_dir: str, overview: str, tree_file: str, out: str) -> None:
    rows = list(csv.DictReader(open(overview), delimiter="\t"))
    by_asm = {r["Assembly name"]: r for r in rows}
    _, tree_names = from_newick(Path(tree_file).read_text())
    in_tree = set(tree_names)

    files = sorted(Path(mca_dir).glob("*.fasta.gz"))
    single = Counter()
    for f in files:
        try:
            with gzip.open(f, "rt") as fh:
                labels = [ln[1:].split()[0] for ln in fh if ln.startswith(">")]
        except (OSError, EOFError):
            continue
        counts = Counter(labels)
        for lab, c in counts.items():
            if c == 1:
                single[lab[3:] if lab.startswith("vs_") else lab] += 1

    with open(out, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["assembly", "species", "common_name", "in_zoonomia_tree", "coverage", "lineage"])
        for asm, n in single.most_common():
            r = by_asm.get(asm)
            sp = r["Species"].replace(" ", "_") if r else ""
            w.writerow([asm, sp, r["Common name"] if r else "", sp in in_tree, round(n / len(files), 4),
                        r["Taxonomic Lineage"] if r else ""])
    print(f"{len(files)} alignments scanned; {len(single)} labels; written {out}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
