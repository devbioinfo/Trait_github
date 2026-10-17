"""Benchmark v2: model-based multi-layer convergence tests under confounding and misspecification.

Scenarios
  baseline     phylogenetic confounding 0/1/2 x effect 0/0.5/0.8/1.2
  heavy_tails  Student-t (df=3) branch noise
  lineage      genome-wide shift on trait lineages (analysed after relative transform)
  tree_error   LMM kernel built from a tree with lognormal-perturbed branch lengths
  sparse/dense signal in 20% / 100% of layers

Usage: python -I scripts/benchmark_v2.py --out results/benchmark_v2.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlconv.lmm import PhyloLMM  # noqa: E402
from mlconv.simulate import LAYERS, simulate_genome  # noqa: E402
from mlconv.stats import bh, relative_transform, run_all_methods, run_lmm_methods  # noqa: E402
from mlconv.tree import (Tree, clade_permute, foreground_branches, kingman_tree,  # noqa: E402
                         origin_sizes, pick_convergent_clades)

BASE = dict(confounding=1.0, noise_df=None, lineage_shift=0.0, p_layer_active=0.6, tree_error=0.0)
SCENARIOS = (
    [("baseline", dict(BASE, confounding=c)) for c in (0.0, 1.0, 2.0)]
    + [("heavy_tails", dict(BASE, noise_df=3.0)),
       ("lineage", dict(BASE, lineage_shift=0.5)),
       ("tree_error", dict(BASE, tree_error=0.5)),
       ("sparse", dict(BASE, p_layer_active=0.2)),
       ("dense", dict(BASE, p_layer_active=1.0))]
)


def layer_auc(layer_post: np.ndarray, active: np.ndarray, called: np.ndarray) -> float:
    sel = called & active.any(1)
    lp, act = layer_post[sel].ravel(), active[sel].ravel()
    if act.all() or not act.any():
        return np.nan
    return mannwhitneyu(lp[act], lp[~act]).statistic / (act.sum() * (~act).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/benchmark_v2.csv")
    ap.add_argument("--tips", type=int, default=100)
    ap.add_argument("--genes", type=int, default=2000)
    ap.add_argument("--null-draws", type=int, default=40)
    ap.add_argument("--perms", type=int, default=300)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--effects", default="0,0.5,0.8,1.2")
    ap.add_argument("--q", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    keep_layers = {"naive:0", "perm:0", "lmm:0"}
    rows = []
    for rep in range(args.reps):
        tree = kingman_tree(args.tips, rng)
        trait = pick_convergent_clades(tree, n_origins=3, size_range=(3, 6), rng=rng)
        fg = foreground_branches(tree, trait)
        nulls = clade_permute(tree, origin_sizes(tree, trait), max(args.perms, args.null_draws),
                              rng, exclude=trait)
        for scen, cfg in SCENARIOS:
            fit_tree = tree
            if cfg["tree_error"] > 0:
                noisy = tree.length * rng.lognormal(0.0, cfg["tree_error"], size=tree.length.size)
                fit_tree = Tree(parent=tree.parent, length=noisy, n_tips=tree.n_tips)
            lmm = PhyloLMM(fit_tree)
            for eff in map(float, args.effects.split(",")):
                t0 = time.time()
                sim = simulate_genome(tree, fg, args.genes, rng, effect=eff,
                                      confounding=cfg["confounding"], noise_df=cfg["noise_df"],
                                      lineage_shift=cfg["lineage_shift"],
                                      p_layer_active=cfg["p_layer_active"])
                X = relative_transform(sim.X)
                truth = sim.is_convergent if eff > 0 else np.zeros(args.genes, dtype=bool)
                pv = run_all_methods(X, fg, nulls[: args.perms])
                pv = {k: v for k, v in pv.items()
                      if k in keep_layers or not (":" in k)}
                pv.update({k: v for k, v in run_lmm_methods(lmm, X, fg, nulls[: args.null_draws]).items()
                           if k in keep_layers or ":" not in k})
                layer_post = pv.pop("_layer_post")
                for name, p in pv.items():
                    rej = bh(p, args.q)
                    rows.append(dict(rep=rep, scenario=scen, effect=eff, method=name,
                                     fpr=(p[~truth] < 0.05).mean(),
                                     fdr=(rej & ~truth).sum() / max(1, rej.sum()),
                                     power=(rej & truth).sum() / max(1, truth.sum()),
                                     layer_auc=layer_auc(layer_post, sim.active_layers, rej)
                                     if name == "ebcm" and eff > 0 else np.nan))
                print(f"rep={rep} {scen} eff={eff} {time.time() - t0:.1f}s", flush=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)       # checkpoint per replicate

    df = pd.DataFrame(rows)
    df["method"] = df["method"].replace({"naive:0": f"naive:{LAYERS[0]}", "perm:0": f"perm:{LAYERS[0]}",
                                         "lmm:0": f"lmm:{LAYERS[0]}"})
    df.to_csv(args.out, index=False)
    for m in ("fpr", "fdr", "power"):
        print(f"\n== {m}")
        print(df.groupby(["scenario", "effect", "method"])[m].mean().round(3).unstack("method").to_string())
    print("\n== EBCM layer attribution AUC")
    print(df[df.method == "ebcm"].groupby(["scenario", "effect"]).layer_auc.mean().round(3).to_string())


if __name__ == "__main__":
    main()
