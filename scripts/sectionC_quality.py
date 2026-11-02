"""Section C quality-rule reconciliation (run on the Mac, where TOGA's MultipleCodonAlignments are stored).

Computes, for the current discordant species and the four candidate additions, the manuscript's species-selection
metric: the fraction of genes in TOGA's multiple codon alignments (human hg38 reference) in which the assembly has
exactly one sequence (1:1 ortholog), as in scripts/toga_species_coverage.py, but without restricting to assemblies in
the Zoonomia tree. It sets this beside the intact share used in results/sectionC/feasibility_toga.csv, so that the two
metrics can be compared (numerator, denominator and treatment of partly intact genes are written to the output).

Gene-level observation states of hypothesis-relevant genes are outcome data and are not computed here; an
unblinding step will be added only after the authors decide to unblind.

Usage: python3 scripts/sectionC_quality.py --mca /path/to/MultipleCodonAlignments
"""
import argparse
import gzip
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPECIES = {"Platanista_gangetica": "HLplaGan1", "Platanista_minor": "HLplaMin1", "Talpa_occidentalis": "HLtalOcc1",
           "Rhizomys_pruinosus": "HLrhiPru1", "Ctenomys_sociabilis": "HLcteSoc1", "Inia_geoffrensis": "HLlniGeo1",
           "Lipotes_vexillifer": "lipVex1", "Ellobius_talpinus": "HLellTal1", "Condylura_cristata": "conCri1",
           "Nannospalax_galili": "nanGal1", "Scalopus_aquaticus": "HLscaAqu1"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mca", required=True, help="folder of TOGA multiple codon alignments (*.fasta.gz)")
    args = ap.parse_args()
    files = sorted(Path(args.mca).glob("*.fasta.gz"))
    if not files:
        raise SystemExit(f"no *.fasta.gz files in {args.mca}")
    single, multi, readable = Counter(), Counter(), 0
    labels_seen = set()
    for f in files:
        try:
            with gzip.open(f, "rt") as fh:
                labs = [ln[1:].split()[0] for ln in fh if ln.startswith(">")]
        except (OSError, EOFError):
            continue
        readable += 1
        c = Counter(lab[3:] if lab.startswith("vs_") else lab for lab in labs)
        labels_seen |= set(c)
        for lab, n in c.items():
            (single if n == 1 else multi)[lab] += 1
    feas = pd.read_csv(ROOT / "results/sectionC/feasibility_toga.csv").set_index("species")
    cov = pd.read_csv(ROOT / "results/toga/coverage_paper.tsv", sep="\t").set_index("assembly").coverage
    rows = []
    for sp, asm in SPECIES.items():
        rows.append(dict(species=sp, assembly=asm, in_alignments=asm in labels_seen,
                         alignment_files_readable=readable,
                         genes_1to1=single[asm], genes_multiple=multi[asm],
                         coverage_1to1=round(single[asm] / readable, 4) if readable else None,
                         coverage_in_coverage_paper_tsv=cov.get(asm),
                         passes_0_85=(single[asm] / readable >= 0.85) if readable else None,
                         intact_share_loss_matrix=feas.intact.get(sp)))
    t = pd.DataFrame(rows)
    out = ROOT / "results/sectionC"
    t.to_csv(out / "quality_reconciliation.csv", index=False)
    (out / "quality_reconciliation_definitions.txt").write_text(
        "coverage_1to1 (manuscript selection metric, scripts/toga_species_coverage.py): numerator = alignment files in which "
        "the assembly has exactly one sequence; denominator = all readable *.fasta.gz alignment files (human hg38 genes). "
        "TOGA's multiple codon alignments contain intact orthologs only, so partly intact or lost genes do not count.\n"
        "intact_share_loss_matrix (results/sectionC/feasibility_toga.csv): numerator = genes of the 115-species loss matrix "
        "whose most intact transcript is I or PI in TOGA's loss_summ_data; denominator = all 19,457 genes of that matrix "
        "(genes without a projection count as missing). Partly intact genes count as intact.\n"
        "Exception: trait species were added by name and need not pass coverage >= 0.85 (scripts/toga_select_species.py); "
        "Ellobius_talpinus (0.835) entered this way.\n")
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
