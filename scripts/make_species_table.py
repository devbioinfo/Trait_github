"""Supplementary Table 3: the 115 species, their genome assemblies and trait coding.

Inputs: results/toga/species_paper.tsv (species, assembly identifier used by TOGA),
results/toga/assemblies_paper.tsv (common name, contig and scaffold N50), results/<trait>_toga/trait_species.txt.
Output: results/extra/supplementary_table3_species.xlsx (and .csv).
Assembly identifiers are those of the TOGA/Zoonomia resources; public accessions (GCA_...) are not in these inputs and
must be added from NCBI before submission (column left empty).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
sp = pd.read_csv(RES / "toga" / "species_paper.tsv", sep="\t", header=None, names=["species", "assembly"])
asm = pd.read_csv(RES / "toga" / "assemblies_paper.tsv", sep="\t").set_index("species")
traits = {"marine": "Marine", "hypoxia_subterranean": "Subterranean", "hypoxia_altitude": "High altitude",
          "echolocation": "Echolocation"}
member = {lab: set((RES / f"{k}_toga" / "trait_species.txt").read_text().split()) for k, lab in traits.items()}
rows = []
for s, a in sp.itertuples(index=False):
    r = asm.loc[s] if s in asm.index else None
    rows.append({"Scientific name": s.replace("_", " "),
                 "Common name": (r["Common name"] if r is not None else ("human" if s == "Homo_sapiens" else "")),
                 "Assembly identifier (TOGA/Zoonomia)": a,
                 "Public accession (to be added)": "",
                 "Contig N50 (bp)": (int(r["contig N50 (bp)"]) if r is not None and pd.notna(r["contig N50 (bp)"]) else None),
                 "Scaffold N50 (bp)": (int(r["scaffold N50 (bp)"]) if r is not None and pd.notna(r["scaffold N50 (bp)"]) else None),
                 **{lab: ("yes" if s in member[lab] else "") for lab in traits.values()}})
t = pd.DataFrame(rows)
t.to_csv(RES / "extra" / "supplementary_table3_species.csv", index=False)
with pd.ExcelWriter(RES / "extra" / "supplementary_table3_species.xlsx") as xw:
    t.to_excel(xw, sheet_name="Supplementary Table 3", index=False, startrow=2)
    ws = xw.sheets["Supplementary Table 3"]
    ws["A1"] = "Supplementary Table 3 | The 115 species, genome assemblies and trait coding"
    for col, w in zip("ABCDEFGHIJ", (28, 26, 22, 22, 14, 14, 10, 12, 12, 12)):
        ws.column_dimensions[col].width = w
    ws.freeze_panes = "C4"
print(len(t), "species;", {k: int((t[k] == "yes").sum()) for k in traits.values()})
