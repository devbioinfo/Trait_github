"""Per-gene evidence for the 63 convergent losses (Supplementary Table 2).

For every called gene and every trait species: TOGA status (L lost, UL uncertain loss,
PI/I intact, M missing) and the inactivating mutations TOGA reports on the orthologous
projection (stop codons, frameshifts, splice-site mutations, deleted exons; mutations TOGA
masks are not counted; missing exons are assembly gaps, not evidence of loss). A loss in a
species is "lesion-supported" if it carries at least one unmasked stop, frameshift or
splice-site mutation or a deleted exon. Per gene and origin we record whether any species
lost the gene and whether any lost species is lesion-supported.

Evidence tiers (rules fixed before looking at the result):
  A  lesion-supported loss in every origin with a loss, and in at least two origins
  B  lesion-supported loss in at least two origins, but some origin's loss lacks a lesion
  C  lesion-supported loss in fewer than two origins

Usage: python -I scripts/loss_evidence.py [--data /home/user/data_toga/detail]
"""
from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from extra_analyses import setup, trait_vec  # noqa: E402
from mlconv.tree import origin_index  # noqa: E402
from toga_loss_matrix import BASE, assembly_dirs, fetch  # noqa: E402

RES = ROOT / "results"
TRAIT_KEY = {"Marine": "marine", "Subterranean": "hypoxia_subterranean"}
LESION = {"STOP": "stop", "FS_DEL": "frameshift", "FS_INS": "frameshift", "SSM": "splice", "Deleted exon": "deleted_exon"}
ORDERS = {"Whippomorpha", "Carnivora", "Afrotheria", "Rodentia", "Eulipotyphla"}


def mutations(path: Path, genes: set[str]) -> pd.DataFrame:
    rows = []
    with gzip.open(path, "rt") as fh:
        for ln in fh:
            f = ln.rstrip("\n").split("\t")
            if len(f) < 8 or not f[0].startswith("# "):
                continue
            sym = f[0][2:].split(".", 1)[1]
            if sym in genes and f[4] in LESION and f[6] != "masked":
                rows.append((sym, LESION[f[4]]))
    return pd.DataFrame(rows, columns=["gene", "kind"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/user/data_toga/detail")
    args = ap.parse_args()
    data = Path(args.data)
    data.mkdir(parents=True, exist_ok=True)
    tree, names, lineage, symbols, Yall = setup()
    status = pd.read_csv(RES / "toga" / "loss_status_paper.tsv.gz", sep="\t", index_col=0)
    calls = pd.read_csv(RES / "extra" / "supplementary_table1_convergent_losses.csv")
    sp = pd.read_csv(RES / "toga" / "species_paper.tsv", sep="\t", header=None, names=["species", "assembly"])
    asm = dict(zip(sp.species, sp.assembly))
    dirs = None
    rows, per_species = [], []
    for trait, grp in calls.groupby("trait", sort=False):
        t = trait_vec(TRAIT_KEY[trait], names, lineage)
        tip_origin = origin_index(tree, t)[: tree.n_tips]
        tsp = [s for s, v in zip(names, t) if v]
        genes = set(grp.gene)
        muts = {}
        for s in tsp:
            dest = data / f"{asm[s]}.geneInactivatingMutations.tsv.gz"
            if not dest.exists():
                dirs = dirs or assembly_dirs(ORDERS)
                fetch(f"{BASE}/{dirs[asm[s]]}/geneInactivatingMutations.tsv.gz", dest.with_suffix(".part"))
                dest.with_suffix(".part").rename(dest)
            muts[s] = mutations(dest, genes)
        for g in grp.gene:
            org_loss, org_lesion = {}, {}
            counts = dict(L=0, UL=0, intact=0, missing=0)
            kinds = dict(stop=0, frameshift=0, splice=0, deleted_exon=0)
            for s in tsp:
                st = status.at[g, s] if g in status.index and s in status.columns else "M"
                o = int(tip_origin[names.index(s)])
                lost = st in ("L", "UL")
                counts["L" if st == "L" else "UL" if st == "UL" else "missing" if st == "M" else "intact"] += 1
                m = muts[s][muts[s].gene == g]
                lesion = lost and len(m) > 0
                for k in m.kind:
                    kinds[k] += 1
                org_loss[o] = org_loss.get(o, False) or lost
                org_lesion[o] = org_lesion.get(o, False) or lesion
                per_species.append(dict(trait=trait, gene=g, species=s, origin=o, status=st,
                                        **{k: int((m.kind == k).sum()) for k in kinds}))
            n_org = len(org_loss)
            n_loss, n_les = sum(org_loss.values()), sum(org_lesion.values())
            tier = "A" if (n_les >= 2 and n_les == n_loss) else "B" if n_les >= 2 else "C"
            rows.append(dict(trait=trait, gene=g, origins=n_org, origins_with_loss=n_loss,
                             origins_lesion_supported=n_les, species_L=counts["L"], species_UL=counts["UL"],
                             species_intact=counts["intact"], species_missing=counts["missing"],
                             **{f"lesions_{k}": v for k, v in kinds.items()}, tier=tier))
    out = pd.DataFrame(rows)
    out.to_csv(RES / "extra" / "loss_evidence_by_gene.csv", index=False)
    pd.DataFrame(per_species).to_csv(RES / "extra" / "loss_evidence_by_species.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))
    print(out.groupby(["trait", "tier"]).size())
    print(out.groupby(["trait", "origins_with_loss"]).size())


if __name__ == "__main__":
    main()
