"""Head-to-head comparison with the RERconverge package on the same AliSim genomes.

For each replicate tree and confounding level, a null genome and convergent genomes
(paired, as in scripts/final_benchmark.py) are simulated and re-estimated. Each gene's
IQ-TREE branch lengths are written as a Newick tree on the true topology, and
RERconverge (getAllResiduals + correlateWithBinaryPhenotype, optionally with binary
permulations) is run on protein trees (its standard input) and on nucleotide trees.
Our final pipeline is run on the same genomes.

RERconverge is called through scripts/run_rerconverge.R. The installed package is
commit 682ac34 of github.com/nclark-lab/RERconverge with scripts/rerconverge_no_ggtree.patch
(ggtree is used only for plotting).

Usage: python -I scripts/rerconverge_compare.py --out results/rerconverge_compare.csv
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from alisim_validation import layers_from_lengths, make_jobs  # noqa: E402
from final_benchmark import EFFECTS  # noqa: E402
from origins_power import estimate  # noqa: E402
from mlconv.gls import learn_covariance, leave_one_origin_out_p  # noqa: E402
from mlconv.stats import acat, bh  # noqa: E402
from mlconv.tree import (foreground_branches, origin_index, pick_convergent_clades,  # noqa: E402
                         to_newick, yule_tree)

HERE = Path(__file__).resolve().parent


def write_trees(path: Path, tree, lengths: np.ndarray) -> None:
    """Gene trees on the true topology; the merged root edge is split evenly."""
    root_kids = np.nonzero(tree.parent[: tree.n_branches] == tree.root)[0]
    with open(path, "w") as fh:
        for g, bl in enumerate(lengths):
            full = np.zeros(tree.n_nodes)
            full[: tree.n_branches] = bl
            full[root_kids] = bl[root_kids[0]] / 2
            fh.write(f"g{g}\t{to_newick(tree, 1.0, full)}\n")


def rerconverge(tree, lengths, trait, n_perms, workdir: Path, tag: str, n_cand: int = 0) -> pd.DataFrame:
    trees_tsv = workdir / f"{tag}_trees.tsv"
    fg_txt = workdir / f"{tag}_fg.txt"
    out_csv = workdir / f"{tag}_rer.csv"
    write_trees(trees_tsv, tree, lengths)
    fg_txt.write_text("\n".join(f"t{i}" for i in np.nonzero(trait)[0]) + "\n")
    root_sp = f"t{int(np.nonzero(~trait)[0][0])}"
    r = subprocess.run(["Rscript", str(HERE / "run_rerconverge.R"), str(trees_tsv), str(fg_txt), root_sp,
                        str(n_perms), str(out_csv), str(n_cand)], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"RERconverge failed ({tag}):\n{r.stderr[-1500:]}")
    df = pd.read_csv(out_csv)
    df["g"] = df.gene.str[1:].astype(int)
    return df.set_index("g").reindex(range(len(lengths)))


def p_from_rer(df: pd.DataFrame, col: str) -> np.ndarray:
    """One-sided p for faster evolution on the foreground; missing genes -> 1.

    Parametric P is two-sided around 0, oriented by Rho. permpvalcor's p is computed on the
    side of the permuted median that the observed Rho falls on, oriented by perm_stat.
    """
    p = df[col].to_numpy(dtype=float)
    sign = (df.perm_stat if col == "perm_p" else df.rho).to_numpy(dtype=float)
    one = np.where(sign > 0, p / 2, 1 - p / 2)
    return np.nan_to_num(one, nan=1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/rerconverge_compare.csv")
    ap.add_argument("--tips", type=int, default=200)
    ap.add_argument("--genes", type=int, default=3000)
    ap.add_argument("--frac", type=float, default=0.02)
    ap.add_argument("--origins", type=int, default=5)
    ap.add_argument("--confounding", default="0,1,2")
    ap.add_argument("--effects", default="moderate")
    ap.add_argument("--perms", type=int, default=0)
    ap.add_argument("--perm-layers", default="protein",
                    help="layers on which RERconverge permulations are run")
    ap.add_argument("--candidates", type=int, default=150,
                    help="permulate only the top-N genes by parametric P (0 = all genes)")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=31)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for rep in range(args.reps):
            tree = yule_tree(args.tips, rng)
            trait = pick_convergent_clades(tree, args.origins, (4, 12), rng)
            fg = foreground_branches(tree, trait)
            origin = origin_index(tree, trait)
            keep = tree.parent[: tree.n_branches] != tree.root
            fk, ok = fg[keep], origin[keep]
            n_conv = int(round(args.frac * args.genes))
            for conf in map(float, args.confounding.split(",")):
                t0 = time.time()
                null_jobs, _, _ = make_jobs(tree, fg, args.genes, rng, frac=0.0, confounding=conf,
                                            omega_mult=1.0, rate_mult=1.0, height=0.8)
                nt0, aa0, cod0 = estimate(null_jobs, args.workers)
                conv_idx = rng.choice(args.genes, size=n_conv, replace=False)
                genomes = [("none", nt0, aa0, cod0, np.zeros(args.genes, bool), None)]
                for eff in args.effects.split(","):
                    om, rm = EFFECTS[eff]
                    cj, _, kinds_c = make_jobs(tree, fg, n_conv, rng, frac=1.0, confounding=conf,
                                               omega_mult=om, rate_mult=rm, height=0.8)
                    ntc, aac, codc = estimate(cj, args.workers)
                    nt, aa, cod = nt0.copy(), aa0.copy(), cod0.copy()
                    nt[conv_idx], aa[conv_idx], cod[conv_idx] = ntc, aac, codc
                    truth = np.zeros(args.genes, bool)
                    truth[conv_idx] = True
                    kinds = np.full(args.genes, "none", dtype=object)
                    kinds[conv_idx] = kinds_c
                    genomes.append((eff, nt, aa, cod, truth, kinds))
                for eff, nt, aa, cod, truth, kinds in genomes:
                    t1 = time.time()
                    X = layers_from_lengths(nt, aa, cod)[:, :, keep]
                    pv = {"final": acat(np.column_stack(
                        [leave_one_origin_out_p(X[:, l], learn_covariance(X[:, l]), ok) for l in range(3)]))}
                    for layer, L in (("protein", aa), ("nucleotide", nt)):
                        n_perm = args.perms if layer in args.perm_layers.split(",") else 0
                        df = rerconverge(tree, L, trait, n_perm, tmp, f"r{rep}c{conf}{eff}{layer}",
                                         args.candidates)
                        pv[f"RERconverge_{layer}"] = p_from_rer(df, "p")
                        if n_perm > 0:
                            pv[f"RERconverge_{layer}_permulation"] = p_from_rer(df, "perm_p")
                    for name, p in pv.items():
                        rej = bh(p, 0.1)
                        row = dict(rep=rep, confounding=conf, effect=eff, method=name,
                                   fpr=(p[~truth] < 0.05).mean(), calls=int(rej.sum()),
                                   false_calls=int((rej & ~truth).sum()),
                                   fdr=(rej & ~truth).sum() / max(1, rej.sum()),
                                   power=(rej & truth).sum() / truth.sum() if truth.any() else np.nan)
                        if kinds is not None:
                            for kind in ("omega", "rate", "both"):
                                sel = kinds == kind
                                row[f"power_{kind}"] = (rej & sel).sum() / max(1, sel.sum())
                        rows.append(row)
                    print(f"rep={rep} conf={conf} {eff}: analysis {time.time() - t1:.0f}s "
                          f"(genome block {time.time() - t0:.0f}s)", flush=True)
                    pd.DataFrame(rows).to_csv(args.out, index=False)

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.groupby(["confounding", "effect", "method"])[["fpr", "false_calls", "fdr", "power"]]
          .mean().round(3).to_string())


if __name__ == "__main__":
    main()
