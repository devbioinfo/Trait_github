"""Gene-level detail for the marine convergent losses (data for Figure 7).

1. funnel   For each marine origin (cetaceans, pinnipeds, sirenians): genes analysed, lost in at
            least one species of the origin, lost in every observed species, of those intact in the
            sister lineage (origin-specific), and genes called convergent (q = 0.1) that are lost in
            the origin. Uses the paper loss matrix (115 species).
2. mutations / exons  TOGA inactivating mutations (stop codons, frameshifts, splice-site mutations,
            deleted or missing exons) and projected exon coordinates in each query genome, for the
            marine calls, in the 18 marine species and 12 terrestrial relatives. Downloaded per
            species from the TOGA hg38 release (geneInactivatingMutations.tsv.gz,
            geneAnnotation.bed.gz) and cached in --data.

Outputs in results/extra/: marine_origin_funnel.csv, marine_detail_mutations.csv,
marine_detail_exons.csv. Usage: python -I scripts/marine_detail.py [--data DIR]
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
import convergence_toga as C  # noqa: E402
from extra_analyses import setup, trait_vec  # noqa: E402
from mlconv.tree import origin_nodes  # noqa: E402
from toga_loss_matrix import BASE, assembly_dirs, fetch  # noqa: E402

RES = ROOT / "results"
OUT = RES / "extra"
RELATIVES = ["Hippopotamus_amphibius", "Bos_taurus", "Sus_scrofa", "Camelus_dromedarius",
             "Pteronura_brasiliensis", "Mellivora_capensis", "Canis_lupus_familiaris", "Felis_catus",
             "Loxodonta_africana", "Procavia_capensis", "Heterohyrax_brucei", "Elephantulus_edwardii"]
ORIGIN_NAMES = {"Cetacea": "Cetaceans", "Otariidae": "Pinnipeds", "Phocidae": "Pinnipeds",
                "Odobenidae": "Pinnipeds", "Sirenia": "Sirenians"}


def funnel(tree, names, lineage, symbols, Yall):
    trait = trait_vec("marine", names, lineage)
    D = tree.tip_sets()
    ok = C.testable(Yall)
    Y, sym = Yall[ok], symbols[ok]
    d = pd.read_csv(RES / "marine_toga" / "marine_genes.csv")
    called = set(d[d.call_loss == True].symbol)  # noqa: E712
    rows = []
    for o in origin_nodes(tree, trait):
        tips = np.nonzero(D[o])[0]
        sister = np.nonzero(D[tree.parent[o]] & ~D[o])[0]
        lin = lineage[names[tips[0]]]
        name = next((v for k, v in ORIGIN_NAMES.items() if k in lin), names[tips[0]])
        yo, ys = Y[:, tips], Y[:, sister]
        any_lost = np.nansum(yo == 1, 1) > 0
        n_obs = (~np.isnan(yo)).sum(1)
        all_lost = any_lost & (np.nansum(yo == 1, 1) == n_obs)
        sister_intact = np.nansum(ys == 1, 1) < np.maximum((~np.isnan(ys)).sum(1), 1)
        specific = all_lost & sister_intact
        conv = np.isin(sym, list(called)) & any_lost
        rows.append(dict(origin=name, species=len(tips), sister_species=len(sister), genes_analysed=len(sym),
                         lost_any=int(any_lost.sum()), lost_all=int(all_lost.sum()),
                         origin_specific=int(specific.sum()), convergent_calls=int(conv.sum()),
                         convergent_lost_all=int((conv & all_lost).sum())))
    return pd.DataFrame(rows)


def read_mutations(path, genes):
    rows = []
    with gzip.open(path, "rt") as fh:
        for ln in fh:
            f = ln.rstrip("\n").split("\t")
            if len(f) < 8 or not f[0].startswith("# "):
                continue
            tx = f[0][2:]
            sym = tx.split(".", 1)[1]
            if sym in genes:
                rows.append(dict(transcript=tx, gene=sym, chain=f[1], exon=int(f[2]), codon=int(f[3]),
                                 type=f[4], detail=f[5], masked=f[6] == "masked", mut_id=f[7]))
    return rows


def read_exons(path, genes):
    rows = []
    with gzip.open(path, "rt") as fh:
        for ln in fh:
            f = ln.rstrip("\n").split("\t")
            parts = f[3].split(".")
            if len(parts) < 3 or parts[1] not in genes:
                continue
            sizes = [int(v) for v in f[10].rstrip(",").split(",")]
            starts = [int(v) for v in f[11].rstrip(",").split(",")]
            for i, (s, z) in enumerate(zip(starts, sizes)):
                rows.append(dict(transcript=".".join(parts[:2]), gene=parts[1], chain=parts[2], chrom=f[0],
                                 strand=f[5], block=i + 1, start=int(f[1]) + s, end=int(f[1]) + s + z,
                                 n_blocks=len(sizes)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/user/data_toga/detail")
    args = ap.parse_args()
    data = Path(args.data)
    data.mkdir(parents=True, exist_ok=True)

    tree, names, lineage, symbols, Yall = setup()
    f = funnel(tree, names, lineage, symbols, Yall)
    f.to_csv(OUT / "marine_origin_funnel.csv", index=False)
    print(f.to_string(index=False))

    d = pd.read_csv(RES / "marine_toga" / "marine_genes.csv")
    genes = set(d[d.call_loss == True].symbol)  # noqa: E712
    marine = [s for s, t in zip(names, trait_vec("marine", names, lineage)) if t]
    sp = pd.read_csv(RES / "toga" / "species_paper.tsv", sep="\t", header=None, names=["species", "assembly"])
    asm = dict(zip(sp.species, sp.assembly))
    orders = {"Whippomorpha", "Ruminantia", "Suina", "Tylopoda", "Carnivora", "Afrotheria"}
    dirs = assembly_dirs(orders)
    muts, exons = [], []
    for s in marine + RELATIVES:
        rel = dirs[asm[s]]
        for fn in ("geneInactivatingMutations.tsv.gz", "geneAnnotation.bed.gz"):
            dest = data / f"{asm[s]}.{fn}"
            if not dest.exists():
                fetch(f"{BASE}/{rel}/{fn}", dest.with_suffix(".part"))
                dest.with_suffix(".part").rename(dest)
        group = "marine" if s in marine else "relative"
        muts += [dict(species=s, group=group, **r) for r in read_mutations(data / f"{asm[s]}.geneInactivatingMutations.tsv.gz", genes)]
        exons += [dict(species=s, group=group, **r) for r in read_exons(data / f"{asm[s]}.geneAnnotation.bed.gz", genes)]
        print("read", s, flush=True)
    pd.DataFrame(muts).to_csv(OUT / "marine_detail_mutations.csv", index=False)
    pd.DataFrame(exons).to_csv(OUT / "marine_detail_exons.csv", index=False)


if __name__ == "__main__":
    main()
