"""Manuscript benchmark: the final pipeline against baselines on AliSim data.

Design (paired): for each replicate tree and confounding level, one null genome is
simulated (AliSim) and re-estimated (IQ-TREE). For each effect size, a fixed set of
genes (--frac of the genome) is replaced by convergent genes simulated on the trait
foreground. Every method sees exactly the same null genes.

Effect sizes (omega multiplier, branch-rate multiplier): weak (1.5, 1.2),
moderate (2.0, 1.35), strong (3.0, 1.5).

Methods
  naive          Welch t of foreground vs background, nucleotide relative rate
                 (RERconverge-like, branches treated as independent)
  permulation    same statistic standardised by control-lineage null foregrounds
                 (RERconverge-like with permulations)
  naive_fisher   Fisher over the three layers of the naive test
  brownian_lmm   phylogenetic LMM with a Brownian kernel, ACAT over layers
  gls            GLS with genome-learned covariance, ACAT over layers
  final          gls + leave-one-origin-out intersection-union test, ACAT over layers
The final pipeline also assigns convergence signatures (learned directions).

Usage: python -I scripts/final_benchmark.py --out results/final_benchmark.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from alisim_validation import LAYERS, layers_from_lengths, make_jobs  # noqa: E402
from origins_power import estimate  # noqa: E402
from signatures import EXPECTED, layer_correlation_from  # noqa: E402
from mlconv.gls import gls_residuals, gls_z, learn_covariance, leave_one_origin_out_p  # noqa: E402
from mlconv.lmm import PhyloLMM  # noqa: E402
from mlconv.signatures import fit_signatures, pattern_active  # noqa: E402
from mlconv.stats import acat, bh, fisher, perm_z, welch_t  # noqa: E402
from mlconv.tree import (foreground_branches, matched_clade_permute, origin_index,  # noqa: E402
                         origin_nodes, pick_convergent_clades, yule_tree)

EFFECTS = {"weak": (1.5, 1.2), "moderate": (2.0, 1.35), "strong": (3.0, 1.5)}


def all_methods(X, fk, ok, lmm, nulls):
    """p-values for every method, plus the inputs the signature model needs."""
    T = welch_t(X, fk[None])[..., 0]
    Tn = welch_t(X[:, :1], nulls)
    p_naive = st.norm.sf(T)
    out = {"naive": p_naive[:, 0],
           "permulation": st.norm.sf(perm_z(T[:, :1], Tn))[:, 0],
           "naive_fisher": fisher(p_naive),
           "brownian_lmm": acat(st.norm.sf(lmm.fit(X, fk).z))}
    z = np.zeros((X.shape[0], X.shape[1]))
    resid = np.zeros_like(X)
    p_looo = []
    for l in range(X.shape[1]):
        S = learn_covariance(X[:, l])
        z[:, l] = gls_z(X[:, l], S, fk[None])[:, 0]
        resid[:, l] = gls_residuals(X[:, l], S, fk)
        p_looo.append(leave_one_origin_out_p(X[:, l], S, ok))
    out["gls"] = acat(st.norm.sf(z))
    out["final"] = acat(np.column_stack(p_looo))
    return out, z, resid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/final_benchmark.csv")
    ap.add_argument("--tips", type=int, default=200)
    ap.add_argument("--genes", type=int, default=3000)
    ap.add_argument("--frac", type=float, default=0.02)
    ap.add_argument("--origins", type=int, default=5)
    ap.add_argument("--min-clade", type=int, default=4)
    ap.add_argument("--max-clade", type=int, default=12)
    ap.add_argument("--confounding", default="0,1,2")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--null-draws", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--discordance", type=float, default=0.0,
                    help="mean nearest-neighbour interchanges per gene tree (0: gene trees = species tree)")
    ap.add_argument("--tree-label", type=int, default=0, help="added to rep in the 'tree' column")
    ap.add_argument("--effects", default="weak,moderate,strong")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    for rep in range(args.reps):
        tree = yule_tree(args.tips, rng)
        trait = pick_convergent_clades(tree, args.origins, (args.min_clade, args.max_clade), rng)
        fg = foreground_branches(tree, trait)
        origin = origin_index(tree, trait)
        keep = tree.parent[: tree.n_branches] != tree.root
        fk, ok = fg[keep], origin[keep]
        nulls = matched_clade_permute(tree, origin_nodes(tree, trait), args.null_draws, rng,
                                      exclude=trait)[:, keep]
        lmm = PhyloLMM(tree, keep=keep)
        n_conv = int(round(args.frac * args.genes))
        for conf in map(float, args.confounding.split(",")):
            t0 = time.time()
            null_jobs, _, _ = make_jobs(tree, fg, args.genes, rng, frac=0.0, confounding=conf,
                                        omega_mult=1.0, rate_mult=1.0, height=0.8, discordance=args.discordance)
            nt0, aa0, cod0 = estimate(null_jobs, args.workers)
            X0 = layers_from_lengths(nt0, aa0, cod0)[:, :, keep]
            p_null, _, _ = all_methods(X0, fk, ok, lmm, nulls)
            conv_idx = rng.choice(args.genes, size=n_conv, replace=False)
            print(f"rep={rep} conf={conf} null genome: {time.time() - t0:.0f}s", flush=True)
            for eff in args.effects.split(","):
                om, rm = EFFECTS[eff]
                t1 = time.time()
                cj, _, kinds_c = make_jobs(tree, fg, n_conv, rng, frac=1.0, confounding=conf,
                                           omega_mult=om, rate_mult=rm, height=0.8, discordance=args.discordance)
                ntc, aac, codc = estimate(cj, args.workers)
                nt, aa, cod = nt0.copy(), aa0.copy(), cod0.copy()
                nt[conv_idx], aa[conv_idx], cod[conv_idx] = ntc, aac, codc
                truth = np.zeros(args.genes, dtype=bool)
                truth[conv_idx] = True
                kinds = np.full(args.genes, "none", dtype=object)
                kinds[conv_idx] = kinds_c
                X = layers_from_lengths(nt, aa, cod)[:, :, keep]
                p_conv, z, resid = all_methods(X, fk, ok, lmm, nulls)

                rej_final = bh(p_conv["final"], 0.1)
                tp = rej_final & truth
                sig_acc = np.nan
                if tp.any():
                    sig = fit_signatures(z, layer_correlation_from(resid), list(LAYERS))
                    act = pattern_active(sig)
                    exp = np.array([EXPECTED[k] for k in kinds[tp]], dtype=bool)
                    sig_acc = (act[tp] == exp).all(1).mean()

                for name in p_conv:
                    rej = bh(p_conv[name], 0.1)
                    row = dict(rep=rep, tree=args.tree_label + rep, tips=args.tips, frac=args.frac,
                               discordance=args.discordance, confounding=conf, effect=eff, method=name,
                               fpr_null=(p_null[name] < 0.05).mean(),
                               false_calls_null=int(bh(p_null[name], 0.1).sum()),
                               fdr=(rej & ~truth).sum() / max(1, rej.sum()),
                               power=(rej & truth).sum() / truth.sum(),
                               signature_accuracy=sig_acc if name == "final" else np.nan)
                    for kind in ("omega", "rate", "both"):
                        sel = kinds == kind
                        row[f"power_{kind}"] = (rej & sel).sum() / max(1, sel.sum())
                    rows.append(row)
                print(f"rep={rep} conf={conf} {eff}: {time.time() - t1:.0f}s", flush=True)
                pd.DataFrame(rows).to_csv(args.out, index=False)

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    for metric in ("fpr_null", "false_calls_null", "fdr", "power"):
        print(f"\n== {metric}")
        print(df.groupby(["confounding", "effect", "method"])[metric].mean().round(3)
              .unstack("method").to_string())
    print("\n== signature accuracy (final)")
    print(df[df.method == "final"].groupby(["confounding", "effect"]).signature_accuracy.mean().round(3))


if __name__ == "__main__":
    main()
