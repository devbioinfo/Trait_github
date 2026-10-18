"""Real-data pilot: echolocation on RERconverge's bundled mammal protein gene trees.

Data: 761 genes present in all 62 species (data/rerconverge_mammals, exported with
scripts/export_rerconverge_trees.R). Trait: laryngeal echolocation in two independent
origins, vespertilionid bats (eptFus1, myoDav1, myoLuc2) and toothed whales (turTru2,
orcOrc1). Pteropodid fruit bats (pteAle1, pteVam1) are non-echolocating.

Layer: log protein branch length (+ a small pseudo-count), mean-centred per branch.

Calibration on real data: the same analysis is repeated for control trait sets, pairs of
non-echolocating clades matched to the observed origins in size and age. Genes called for a
control set are false calls by construction. Methods: naive Welch t (RERconverge-like,
branches independent), parametric RERconverge (via scripts/rer_foregrounds.R), learned-
covariance GLS, and the final test (GLS + leave-one-origin-out).

Usage: python -I scripts/echolocation_pilot.py --outdir results/echolocation_pilot
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mlconv.gls import gls_z, learn_covariance, leave_one_origin_out_p  # noqa: E402
from mlconv.stats import bh, welch_t  # noqa: E402
from mlconv.tree import (branch_splits, foreground_branches, from_newick,  # noqa: E402
                         matched_clade_permute, named_splits, origin_index, origin_nodes)

ECHOLOCATORS = ["eptFus1", "myoDav1", "myoLuc2", "turTru2", "orcOrc1"]


def load(data_dir: Path, pseudo: float):
    tree, names = from_newick((data_dir / "master.nwk").read_text())
    index = {n: i for i, n in enumerate(names)}
    keys = branch_splits(tree)
    genes, rows = [], []
    for line in (data_dir / "gene_trees.tsv").read_text().splitlines():
        gene, nwk = line.split("\t")
        sp = named_splits(nwk, index, tree.n_tips)
        rows.append([sp.get(k, 0.0) for k in keys])
        genes.append(gene)
    bl = np.array(rows)
    keep = tree.parent[: tree.n_branches] != tree.root        # root edges are not identifiable
    X = np.log(bl[:, keep] + pseudo)
    return tree, names, genes, X - X.mean(0, keepdims=True), keep, bl


def our_methods(X, fk, ok):
    S = learn_covariance(X)
    return {"naive": st.norm.sf(welch_t(X[:, None, :], fk[None])[:, 0, 0]),
            "gls": st.norm.sf(gls_z(X, S, fk[None])[:, 0]),
            "final": leave_one_origin_out_p(X, S, ok)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "rerconverge_mammals"))
    ap.add_argument("--outdir", default=str(ROOT / "results" / "echolocation_pilot"))
    ap.add_argument("--controls", type=int, default=100)
    ap.add_argument("--pseudo", type=float, default=1e-3)
    ap.add_argument("--q", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    tree, names, genes, X, keep, bl = load(Path(args.data), args.pseudo)
    trait = np.isin(names, ECHOLOCATORS)
    fg = foreground_branches(tree, trait)
    origins = origin_nodes(tree, trait)
    origin = origin_index(tree, trait)
    print(f"{len(genes)} genes, {tree.n_tips} species, {int(keep.sum())} branches analysed; "
          f"{len(origins)} origins, {int(fg[keep].sum())} foreground branches")

    # control trait sets: matched in clade size and age, no echolocators
    ctrl = matched_clade_permute(tree, origins, 5 * args.controls, rng, exclude=trait, size_tol=1)
    ctrl = np.unique(ctrl, axis=0)
    ctrl = ctrl[[not np.array_equal(c, fg) for c in ctrl]]
    D = tree.tip_sets()

    def n_origins(s):                                      # sister control clades merge into one
        tips = np.array([any(D[b, t] for b in np.nonzero(s)[0]) for t in range(tree.n_tips)])
        return len(origin_nodes(tree, tips))
    ctrl = np.array([c for c in ctrl if n_origins(c) == len(origins)])[: args.controls]
    sets = [fg] + list(ctrl)
    tip_sets = [[names[t] for t in range(tree.n_tips) if any(D[b, t] for b in np.nonzero(s)[0])] for s in sets]
    print(f"{len(ctrl)} distinct control trait sets")

    # RERconverge, all trait sets at once
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "fg.txt").write_text("\n".join(",".join(t) for t in tip_sets) + "\n")
        r = subprocess.run(["Rscript", str(ROOT / "scripts" / "rer_foregrounds.R"),
                            str(Path(args.data) / "gene_trees.tsv"), str(tmp / "fg.txt"),
                            str(tmp / "rho.csv"), str(tmp / "p.csv")], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr[-2000:])
        rho = pd.read_csv(tmp / "rho.csv", index_col=0).reindex(genes)
        prr = pd.read_csv(tmp / "p.csv", index_col=0).reindex(genes)

    def rer_p(i):
        p, r_ = prr.iloc[:, i].to_numpy(float), rho.iloc[:, i].to_numpy(float)
        return np.nan_to_num(np.where(r_ > 0, p / 2, 1 - p / 2), nan=1.0)

    rows = []
    results = {}
    for i, s in enumerate(sets):
        trait_i = np.zeros(tree.n_tips, bool)
        trait_i[[names.index(n) for n in tip_sets[i]]] = True
        oi = origin_index(tree, trait_i)[keep]
        pv = our_methods(X, s[keep], oi)
        pv["RERconverge"] = rer_p(i)
        if i == 0:
            results = pv
        for name, p in pv.items():
            rows.append(dict(set="echolocation" if i == 0 else f"control_{i}", method=name,
                             calls=int(bh(p, args.q).sum()), min_p=float(p.min()),
                             species=";".join(tip_sets[i])))
    calls = pd.DataFrame(rows)
    calls.to_csv(out / "calls_per_trait_set.csv", index=False)

    ctrl_calls = calls[calls.set != "echolocation"]
    summ = ctrl_calls.groupby("method").calls.agg(
        mean_false_calls="mean", sets_with_any_call=lambda c: (c > 0).mean(), max_calls="max")
    summ["echolocation_calls"] = calls[calls.set == "echolocation"].set_index("method").calls
    summ.to_csv(out / "calibration_summary.csv")
    print("\nCalls at q = %.2f: echolocation vs %d matched control trait sets" % (args.q, len(ctrl)))
    print(summ.round(3).to_string())

    top = pd.DataFrame({"gene": genes, **{f"p_{k}": v for k, v in results.items()}})
    for k in results:
        top[f"q_{k}"] = st.false_discovery_control(np.clip(results[k], 1e-300, 1))
    top = top.sort_values("p_final")
    top.to_csv(out / "echolocation_genes.csv", index=False)
    print("\nTop genes by the final test:")
    print(top.head(15).to_string(index=False, float_format=lambda v: f"{v:.2e}"))
    print("\nCDH23:", top[top.gene == "CDH23"].to_string(index=False, float_format=lambda v: f"{v:.3g}"))


if __name__ == "__main__":
    main()
