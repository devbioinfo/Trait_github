"""Codon alignments of two convergent losses for Figure 8.

PON1 (marine) in the 18 marine species and 12 terrestrial relatives; LGSN (subterranean) in the 8
subterranean species and 7 relatives. TOGA's multiple codon alignments contain intact orthologs
only, so the pairwise human-query codon alignments (codonAlignments.fa.gz) of each species are
streamed from the TOGA hg38 release and only the two genes are kept. For each species the
projection (chain) with the most aligned codons is used. Query codons opposite reference gaps
(insertions) are dropped; codons TOGA marks as disrupted by a frameshift are kept as "XXX".

Output: results/extra/codon_alignment_detail.csv (species, gene, chain, ref_pos, ref, query).
Usage: python -I scripts/alignment_detail.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from marine_detail import RELATIVES  # noqa: E402
from toga_loss_matrix import BASE, assembly_dirs  # noqa: E402

RES = ROOT / "results"
OUT = RES / "extra" / "codon_alignment_detail.csv"
SUBTERRANEAN_RELATIVES = ["Mus_musculus", "Rattus_norvegicus", "Microtus_ochrogaster", "Dinomys_branickii",
                          "Thryonomys_swinderianus", "Erinaceus_europaeus", "Elephantulus_edwardii"]
ORDERS = {"Whippomorpha", "Ruminantia", "Suina", "Tylopoda", "Carnivora", "Afrotheria", "Rodentia",
          "Eulipotyphla"}


def stream_gene(url: str, genes: set[str], tries: int = 3) -> dict[tuple[str, str], tuple[list, list]]:
    """(gene, chain) -> (reference codons, query codons) from a gzipped pairwise codon alignment."""
    for k in range(tries):
        p = subprocess.Popen(f"curl -sfL --max-time 1800 '{url}' | zcat", shell=True, stdout=subprocess.PIPE, text=True)
        recs, cur, keep = {}, None, False
        for ln in p.stdout:
            if ln.startswith(">"):
                tx = ln[1:].split(" | ")[0]
                parts = tx.split(".")
                keep = len(parts) >= 3 and parts[1] in genes
                if keep:
                    cur = (parts[1], parts[2], "REFERENCE" in ln)
                    recs.setdefault(cur, [])
            elif keep:
                recs[cur].append(ln.strip())
        if p.wait() == 0 or recs:
            out = {}
            for (g, ch, is_ref), seq in recs.items():
                if is_ref:
                    out[(g, ch)] = (" ".join(seq).split(), " ".join(recs.get((g, ch, False), [])).split())
            return out
    raise RuntimeError(f"download failed: {url}")


def main():
    sets = {"PON1": [l.strip() for l in open(RES / "marine_toga" / "trait_species.txt") if l.strip()] + RELATIVES,
            "LGSN": [l.strip() for l in open(RES / "hypoxia_subterranean_toga" / "trait_species.txt") if l.strip()]
            + SUBTERRANEAN_RELATIVES}
    sp = pd.read_csv(RES / "toga" / "species_paper.tsv", sep="\t", header=None, names=["species", "assembly"])
    asm = dict(zip(sp.species, sp.assembly))
    dirs = assembly_dirs(ORDERS)
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["species", "gene"])
    rows = [done]
    for s in sorted({s for v in sets.values() for s in v}):
        genes = {g for g, v in sets.items() if s in v}
        if done[(done.species == s)].gene.isin(genes).sum():
            continue
        recs = stream_gene(f"{BASE}/{dirs[asm[s]]}/codonAlignments.fa.gz", genes)
        for g in genes:
            cands = {ch: rq for (gg, ch), rq in recs.items() if gg == g}
            if not cands:
                continue
            ch, (r, q) = max(cands.items(), key=lambda kv: sum(c != "---" for c in kv[1][1]))
            pos, recs_g = 0, []
            for a, b in zip(r, q):
                if a == "---":
                    continue
                pos += 1
                recs_g.append(dict(species=s, gene=g, chain=ch, ref_pos=pos, ref=a, query=b))
            rows.append(pd.DataFrame(recs_g))
        pd.concat(rows, ignore_index=True).to_csv(OUT, index=False)
        print("done", s, sorted(genes), flush=True)


if __name__ == "__main__":
    main()
