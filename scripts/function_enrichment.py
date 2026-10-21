"""Tissue, pathway and Gene Ontology enrichment of the convergent gene losses.

Annotations (downloaded by the user and placed in --annot):
  rna_tissue_consensus.tsv   Human Protein Atlas consensus nTPM, 50 tissues
  ReactomePathways.gmt       Reactome human pathways (gene symbols)
  goa_human.gaf.gz + go-basic.obo   GO annotations and ontology (biological process; optional)

A gene is "elevated" in a tissue when its nTPM there is >= 1 and >= 4 times its mean over all
tissues (after the Human Protein Atlas tissue-enhanced rule). GO annotations are propagated to
ancestor terms through is_a and part_of.

Background: genes tested by the loss layer for the trait. Genes that are often lost across all
mammals (olfactory receptors, tissue-specific genes) are over-represented among any set of loss
calls, so the null keeps that frequency: every called gene is replaced by a random tested gene
from the same bin of total losses across the 115 species (20 quantile bins). The number of null
genes in a term is then a sum of independent Bernoulli variables (one per called gene, with the
term's share of that gene's bin), and the p-value is its exact upper tail (Poisson-binomial); BH
within each annotation source and trait over all terms of testable size (5-1,000 genes); terms with
fewer than two called genes are not tested and enter the correction with p = 1.

Outputs: results/extra/enrichment_{tissue,reactome,go}.csv
Usage: python -I scripts/function_enrichment.py [--annot DIR]
"""
from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "extra"
TRAITS = ["marine", "hypoxia_subterranean"]


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def tissue_sets(annot):
    t = pd.read_csv(annot / "rna_tissue_consensus.tsv", sep="\t")
    m = t.pivot_table(index="Gene name", columns="Tissue", values="nTPM", aggfunc="max")
    elevated = (m >= 1) & (m.ge(4 * m.mean(axis=1), axis=0))
    return {tis: set(elevated.index[elevated[tis]]) for tis in m.columns}, set(m.index)


def reactome_sets(annot):
    sets, names = {}, {}
    for ln in open(annot / "ReactomePathways.gmt"):
        f = ln.rstrip("\n").split("\t")
        sets[f[1]] = set(f[2:])
        names[f[1]] = f[0]
    return sets, names


def go_sets(annot):
    obo = annot / "go-basic.obo"
    if not obo.exists():
        return None, None
    names, parents, space, cur = {}, {}, {}, None
    for ln in open(obo):
        ln = ln.rstrip("\n")
        if ln == "[Term]":
            cur = None
        elif ln.startswith("id: GO:"):
            cur = ln[4:]
            parents[cur] = set()
        elif cur and ln.startswith("name: "):
            names[cur] = ln[6:]
        elif cur and ln.startswith("namespace: "):
            space[cur] = ln[11:]
        elif cur and ln.startswith("is_a: "):
            parents[cur].add(ln[6:16])
        elif cur and ln.startswith("relationship: part_of "):
            parents[cur].add(ln[22:32])
    anc = {}

    def ancestors(t):
        if t not in anc:
            a = {t}
            for p in parents.get(t, ()):
                a |= ancestors(p)
            anc[t] = a
        return anc[t]

    sets = {}
    with gzip.open(annot / "goa_human.gaf.gz", "rt") as fh:
        for ln in fh:
            if ln.startswith("!"):
                continue
            f = ln.split("\t")
            if f[8] != "P" or "NOT" in f[3]:
                continue
            for t in ancestors(f[4]):
                if space.get(t) == "biological_process":
                    sets.setdefault(t, set()).add(f[2])
    return sets, names


def poisson_binomial_tail(probs, k):
    """P(sum of independent Bernoulli(probs) >= k), exact by convolution."""
    dist = np.zeros(len(probs) + 1)
    dist[0] = 1.0
    for q in probs:
        dist[1:] = dist[1:] * (1 - q) + dist[:-1] * q
        dist[0] *= 1 - q
    return float(dist[k:].sum())


def enrich(sets, names, background, called, bins, min_size=5, max_size=1000):
    genes = np.array(sorted(background))
    idx = {g: i for i, g in enumerate(genes)}
    keep = {t: s & background for t, s in sets.items()}
    keep = {t: s for t, s in keep.items() if min_size <= len(s) <= max_size}
    n_eligible = len(keep)                 # BH counts every term of testable size
    keep = {t: s for t, s in keep.items() if len(s & called) >= 2}
    if not keep:
        return pd.DataFrame()
    terms = list(keep)
    M = np.zeros((len(terms), len(genes)), np.float32)
    for i, t in enumerate(terms):
        M[i, [idx[g] for g in keep[t]]] = 1
    cidx = np.array([idx[g] for g in sorted(called)])
    obs = M[:, cidx].sum(1)
    b = np.array([bins[g] for g in genes])
    ub = np.unique(b)
    share = np.stack([M[:, b == v].mean(1) for v in ub], 1)       # terms x bins: share of bin genes in term
    probs = share[:, np.searchsorted(ub, b[cidx])]                  # terms x called genes
    null_mean = probs.sum(1)
    p = np.array([poisson_binomial_tail(probs[i], int(obs[i])) for i in range(len(terms))])
    df = pd.DataFrame({"term": terms, "name": [names.get(t, t) for t in terms], "size": M.sum(1).astype(int),
                       "called_in_term": obs.astype(int), "called": len(cidx),
                       "expected_matched": null_mean, "p": p,
                       "genes": [";".join(sorted(keep[t] & called)) for t in terms]})
    df["fold"] = df.called_in_term / df.expected_matched.clip(lower=1e-9)
    df["q"] = bh(np.concatenate([df.p.to_numpy(), np.ones(n_eligible - len(df))]))[: len(df)]
    df["terms_eligible"] = n_eligible
    return df.sort_values("p")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annot", default="/home/user/data_toga/annot")
    args = ap.parse_args()
    annot = Path(args.annot)
    m = pd.read_csv(RES / "toga" / "loss_status_paper.tsv.gz", sep="\t", index_col=0)
    total = m.isin(["L", "UL"]).sum(axis=1)
    tis, hpa_genes = tissue_sets(annot)
    rea, rea_names = reactome_sets(annot)
    go, go_names = go_sets(annot)
    sources = [("tissue", tis, {t: t for t in tis}), ("reactome", rea, rea_names)]
    if go is not None:
        sources.append(("go", go, go_names))
    for name, sets, names in sources:
        rows = []
        for k in TRAITS:
            d = pd.read_csv(RES / f"{k}_toga" / f"{k}_genes.csv").dropna(subset=["p_loss"]).drop_duplicates("symbol")
            background = set(d.symbol)
            if name == "tissue":
                background &= hpa_genes
            called = set(d[d.call_loss == True].symbol) & background  # noqa: E712
            tb = total.reindex(sorted(background)).fillna(0)
            bins = pd.Series(pd.qcut(tb.rank(method="first"), 20, labels=False).to_numpy(), index=tb.index)
            r = enrich(sets, names, background, called, bins)
            if len(r):
                r.insert(0, "trait", k)
                r.insert(1, "background", len(background))
                rows.append(r)
            print(name, k, f"{len(called)} called of {len(background)};",
                  "q<0.1:", ", ".join(r[r.q < 0.1].name.head(8)) if len(r) else "-", flush=True)
        pd.concat(rows).to_csv(OUT / f"enrichment_{name}.csv", index=False)


if __name__ == "__main__":
    main()
