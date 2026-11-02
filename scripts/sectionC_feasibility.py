"""Section C feasibility: TOGA annotation completeness and ocular-gene coverage for candidate additions.

For each candidate assembly, downloads TOGA's loss_summ_data.tsv.gz (hg38 reference), takes each gene's most
intact transcript (as scripts/toga_loss_matrix.py) and reports, over the genes of the 115-species loss matrix:
shares classified intact (I/PI), uncertain loss (UL), lost (L) and missing (M or not projected), with the
percentile of the intact share among the 114 non-reference species already analysed. It also lists the status of
the ocular genes of Fig. 6a and of the 15 subterranean candidates in each candidate and in selected current species.
Output: results/sectionC/feasibility_toga.csv, results/sectionC/feasibility_ocular_genes.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import toga_loss_matrix as tlm  # noqa: E402

CANDIDATES = {"Platanista_gangetica": "HLplaGan1", "Platanista_minor": "HLplaMin1",
              "Talpa_occidentalis": "HLtalOcc1", "Rhizomys_pruinosus": "HLrhiPru1"}
CURRENT = ["Inia_geoffrensis", "Lipotes_vexillifer", "Ctenomys_sociabilis", "Nannospalax_galili",
           "Condylura_cristata", "Scalopus_aquaticus"]
EYE = ["CRYAA", "CRYBA1", "CRYBA4", "CRYBB2", "CRYGC", "CRYGS", "BFSP1", "BFSP2", "LGSN", "MIP", "GJA3", "GJA8",
       "LCTL", "OPTC", "RHO", "GNAT1", "PDE6A", "PDE6B", "CNGA1", "PRPH2", "ROM1", "MPP4", "GRM6", "GNAO1", "TRPM1",
       "RGS11", "RGS7", "GNB5", "NYX", "CABP5"]
OUT = ROOT / "results" / "sectionC"


def shares(s):
    s = s.fillna("M")
    return {k: float((s == k).mean()) for k in ("I", "PI", "UL", "L", "M")}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "data" / "toga_cache_sectionC"
    cache.mkdir(parents=True, exist_ok=True)
    mat = pd.read_csv(ROOT / "results/toga/loss_status_paper.tsv.gz", sep="\t", index_col=0)
    genes = mat.index
    subt = pd.read_csv(ROOT / "results/hypoxia_subterranean_toga/hypoxia_subterranean_genes.csv")
    cand15 = sorted(subt[subt.call_loss == True].symbol)  # noqa: E712
    top = set(__import__("re").findall(r'href="([A-Z][A-Za-z]+)/"', tlm.fetch(f"{tlm.BASE}/"))) - {"Matrix", "MultipleCodonAlignments"}
    dirs = tlm.assembly_dirs(top)
    new = {}
    for sp, asm in CANDIDATES.items():
        f = cache / f"{asm}.loss_summ_data.tsv.gz"
        if not f.exists():
            tlm.fetch(f"{tlm.BASE}/{dirs[asm]}/loss_summ_data.tsv.gz", f)
        new[sp] = tlm.gene_status(f).reindex(genes)
    # reference distribution of the intact share among the current species
    cur = {sp: shares(mat[sp]) for sp in mat.columns if sp != "Homo_sapiens"}
    intact_cur = np.array([v["I"] + v["PI"] for v in cur.values()])
    rows = []
    for sp, s in list(new.items()) + [(sp, mat[sp]) for sp in CURRENT]:
        sh = shares(s)
        it = sh["I"] + sh["PI"]
        rows.append(dict(species=sp, assembly=CANDIDATES.get(sp, "current set"), genes=len(genes),
                         intact=round(it, 4), uncertain_loss=round(sh["UL"], 4), lost=round(sh["L"], 4),
                         missing=round(sh["M"], 4),
                         intact_percentile_among_current=round(float((intact_cur < it).mean() * 100), 1),
                         ocular_genes_with_data=int(s.reindex(EYE).fillna("M").ne("M").sum()),
                         candidates15_with_data=int(s.reindex(cand15).fillna("M").ne("M").sum())))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "feasibility_toga.csv", index=False)
    print(t.to_string(index=False))
    print(f"\ncurrent species intact share: min {intact_cur.min():.3f}, 5th pct {np.percentile(intact_cur, 5):.3f}, "
          f"median {np.median(intact_cur):.3f}")
    # Only data availability is reported. Per-gene loss states are outcome data and are not written or printed
    # at the feasibility stage, so that phenotype coding stays blind to them.


if __name__ == "__main__":
    main()
