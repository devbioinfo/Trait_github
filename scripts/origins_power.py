"""Power and error rates as a function of the number of independent origins.

Per replicate tree (Yule, --tips species), six non-overlapping trait clades are drawn.
A null genome (no convergence) is simulated once with AliSim and re-estimated with
IQ-TREE; it does not depend on which clades carry the trait. For m = 2..6 origins the
trait is the union of the first m clades, and a fixed set of genes (--frac of the
genome) is replaced by convergent genes simulated on that foreground. The comparison
across m is therefore paired: same tree, same null genes, nested origins.

Usage: python -I scripts/origins_power.py --out results/origins_power.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from alisim_validation import LAYERS, layers_from_lengths, make_jobs, simulate_and_estimate  # noqa: E402
from mlconv.gls import gls_z, learn_covariance, leave_one_origin_out_p  # noqa: E402
from mlconv.stats import acat, bh, welch_t  # noqa: E402
from mlconv.tree import foreground_branches, origin_index, origin_nodes, pick_convergent_clades, yule_tree  # noqa: E402


def estimate(jobs, workers):
    with Pool(workers) as pool:
        est = pool.map(simulate_and_estimate, jobs, chunksize=8)
    return (np.array([e[0] for e in est]), np.array([e[1] for e in est]),
            np.array([j["codons"] for j in jobs]))


def methods(X: np.ndarray, fg: np.ndarray, origin: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    p_pool, p_looo = [], []
    for l, name in enumerate(LAYERS):
        S = learn_covariance(X[:, l])
        p_pool.append(st.norm.sf(gls_z(X[:, l], S, fg[None])[:, 0]))
        p_looo.append(leave_one_origin_out_p(X[:, l], S, origin))
        out[f"gls:{name}"] = p_pool[-1]
        out[f"gls_looo:{name}"] = p_looo[-1]
    out["gls_acat"] = acat(np.column_stack(p_pool))
    out["gls_looo_acat"] = acat(np.column_stack(p_looo))
    out["naive:rate_nucleotide"] = st.norm.sf(welch_t(X[:, :1], fg[None])[:, 0, 0])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/origins_power.csv")
    ap.add_argument("--tips", type=int, default=200)
    ap.add_argument("--genes", type=int, default=3000)
    ap.add_argument("--frac", type=float, default=0.02)
    ap.add_argument("--max-origins", type=int, default=6)
    ap.add_argument("--min-clade", type=int, default=4)
    ap.add_argument("--max-clade", type=int, default=12)
    ap.add_argument("--confounding", type=float, default=1.0)
    ap.add_argument("--omega-mult", type=float, default=3.0)
    ap.add_argument("--rate-mult", type=float, default=1.5)
    ap.add_argument("--height", type=float, default=0.8)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--rep-offset", type=int, default=0, help="added to rep in the output")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    for rep in range(args.reps):
        t0 = time.time()
        tree = yule_tree(args.tips, rng)
        trait_all = pick_convergent_clades(tree, args.max_origins, (args.min_clade, args.max_clade), rng)
        clades = origin_nodes(tree, trait_all)
        rng.shuffle(clades)
        D = tree.tip_sets()
        keep = tree.parent[: tree.n_branches] != tree.root
        dummy_fg = foreground_branches(tree, trait_all)
        null_jobs, _, _ = make_jobs(tree, dummy_fg, args.genes, rng, frac=0.0, confounding=args.confounding,
                                    omega_mult=args.omega_mult, rate_mult=args.rate_mult, height=args.height)
        nt0, aa0, cod0 = estimate(null_jobs, args.workers)
        n_conv = int(round(args.frac * args.genes))
        conv_idx = rng.choice(args.genes, size=n_conv, replace=False)
        print(f"rep={rep} null genome: {time.time() - t0:.0f}s", flush=True)
        for m in range(2, args.max_origins + 1):
            t1 = time.time()
            trait = np.zeros(tree.n_tips, dtype=bool)
            for v in clades[:m]:
                trait |= D[v]
            fg = foreground_branches(tree, trait)
            origin = origin_index(tree, trait)
            conv_jobs, _, kinds_c = make_jobs(tree, fg, n_conv, rng, frac=1.0, confounding=args.confounding,
                                              omega_mult=args.omega_mult, rate_mult=args.rate_mult,
                                              height=args.height)
            ntc, aac, codc = estimate(conv_jobs, args.workers)
            nt, aa, cod = nt0.copy(), aa0.copy(), cod0.copy()
            nt[conv_idx], aa[conv_idx], cod[conv_idx] = ntc, aac, codc
            truth = np.zeros(args.genes, dtype=bool)
            truth[conv_idx] = True
            kinds = np.full(args.genes, "none", dtype=object)
            kinds[conv_idx] = kinds_c
            fk, ok = fg[keep], origin[keep]
            p_null = methods(layers_from_lengths(nt0, aa0, cod0)[:, :, keep], fk, ok)
            p_conv = methods(layers_from_lengths(nt, aa, cod)[:, :, keep], fk, ok)
            for name in p_conv:
                rej_n = bh(p_null[name], 0.1)
                rej = bh(p_conv[name], 0.1)
                row = dict(rep=rep + args.rep_offset, origins=m, n_fg_branches=int(fk.sum()), method=name,
                           fpr_null=(p_null[name] < 0.05).mean(), false_calls_null=int(rej_n.sum()),
                           fdr=(rej & ~truth).sum() / max(1, rej.sum()),
                           power=(rej & truth).sum() / max(1, truth.sum()))
                for kind in ("omega", "rate", "both"):
                    sel = kinds == kind
                    row[f"power_{kind}"] = (rej & sel).sum() / max(1, sel.sum())
                rows.append(row)
            print(f"rep={rep} origins={m} fg_branches={fk.sum()}: {time.time() - t1:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(args.out, index=False)

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    for metric in ("fpr_null", "false_calls_null", "fdr", "power"):
        print(f"\n== {metric}")
        print(df.groupby(["origins", "method"])[metric].mean().round(3).unstack("origins").to_string())


if __name__ == "__main__":
    main()
