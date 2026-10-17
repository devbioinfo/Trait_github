"""Benchmark: calibration and power of single-layer and multi-layer convergence tests.

Usage: python -I scripts/benchmark.py --out results/benchmark.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlconv.simulate import LAYERS, simulate_genome  # noqa: E402
from mlconv.stats import bh, run_all_methods  # noqa: E402
from mlconv.tree import (clade_permute, foreground_branches, kingman_tree, origin_sizes,  # noqa: E402
                         permulate, pick_convergent_clades)


def evaluate(pvals: dict[str, np.ndarray], truth: np.ndarray, alpha: float, q: float) -> list[dict]:
    rows = []
    for name, p in pvals.items():
        rej = bh(p, q)
        tp = (rej & truth).sum()
        fp = (rej & ~truth).sum()
        rows.append(dict(method=name,
                         fpr=(p[~truth] < alpha).mean(),
                         fdr=fp / max(1, rej.sum()),
                         power=tp / max(1, truth.sum())))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/benchmark.csv")
    ap.add_argument("--tips", type=int, default=100)
    ap.add_argument("--genes", type=int, default=2000)
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--effects", default="0.0,0.3,0.5,0.8")
    ap.add_argument("--confounding", default="0,0.5,1,2")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--null", choices=["clade", "bm"], default="clade")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    for rep in range(args.reps):
        tree = kingman_tree(args.tips, rng)
        trait = pick_convergent_clades(tree, n_origins=3, size_range=(3, 6), rng=rng)
        fg = foreground_branches(tree, trait)
        if args.null == "clade":
            perms = clade_permute(tree, origin_sizes(tree, trait), args.perms, rng, exclude=trait)
        else:
            perms = permulate(tree, fg, args.perms, rng)
        for conf in map(float, args.confounding.split(",")):
            for eff in map(float, args.effects.split(",")):
                t0 = time.time()
                sim = simulate_genome(tree, fg, args.genes, rng, effect=eff, confounding=conf)
                truth = sim.is_convergent if eff > 0 else np.zeros(args.genes, dtype=bool)
                res = evaluate(run_all_methods(sim.X, fg, perms), truth, alpha=0.05, q=0.1)
                for r in res:
                    r.update(rep=rep, confounding=conf, effect=eff, n_fg=int(fg.sum()))
                rows += res
                print(f"rep={rep} conf={conf} eff={eff} fg={fg.sum()} {time.time() - t0:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    names = {f"naive:{i}": f"naive:{l}" for i, l in enumerate(LAYERS)}
    names.update({f"perm:{i}": f"perm:{l}" for i, l in enumerate(LAYERS)})
    df["method"] = df["method"].replace(names)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    summary = (df.groupby(["confounding", "effect", "method"])[["fpr", "fdr", "power"]]
               .mean().round(3))
    print(summary.to_string())


if __name__ == "__main__":
    main()
