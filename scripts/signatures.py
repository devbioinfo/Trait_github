"""Convergence signatures: EBCM on learned-covariance GLS z-scores (AliSim data).

Detection uses the validated test (GLS with genome-learned covariance, leave-one-origin-out,
ACAT over layers). EBCM is fitted on the pooled GLS z-scores with the cross-layer
correlation of GLS residuals, and its per-layer posteriors are compared with the layers
each kind of simulated convergence should activate:

    omega shift (selection):  rate_nucleotide 0, rate_protein 1, selection_proxy 1
    rate shift:               rate_nucleotide 1, rate_protein 1, selection_proxy 0
    both:                     1, 1, 1

Usage: python -I scripts/signatures.py --out results/signatures.csv
"""
from __future__ import annotations

import argparse
import json
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
from mlconv.configs import fit_configurations  # noqa: E402
from mlconv.gls import gls_residuals, gls_z, learn_covariance, leave_one_origin_out_p  # noqa: E402
from mlconv.stats import acat, bh  # noqa: E402
from mlconv.tree import foreground_branches, origin_index, pick_convergent_clades, yule_tree  # noqa: E402

EXPECTED = {"omega": (0, 1, 1), "rate": (1, 1, 0), "both": (1, 1, 1)}


def layer_correlation_from(resid: np.ndarray, shrink: float = 0.1) -> np.ndarray:
    """Per-gene cross-layer correlation of whitened residuals (G, L, B), shrunk to the mean."""
    Z = resid - resid.mean(-1, keepdims=True)
    C = np.einsum("glb,gkb->glk", Z, Z)
    sd = np.sqrt(np.einsum("gll->gl", C))
    R = C / sd[:, :, None] / sd[:, None, :]
    R = (1 - shrink) * R + shrink * R.mean(0, keepdims=True)
    return R + 1e-3 * np.eye(R.shape[1])[None]


def auc(score: np.ndarray, label: np.ndarray) -> float:
    if label.all() or not label.any():
        return float("nan")
    return float(st.mannwhitneyu(score[label], score[~label]).statistic / (label.sum() * (~label).sum()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/signatures.csv")
    ap.add_argument("--tips", type=int, default=200)
    ap.add_argument("--genes", type=int, default=3000)
    ap.add_argument("--frac", type=float, default=0.02)
    ap.add_argument("--origins", type=int, default=5)
    ap.add_argument("--min-clade", type=int, default=4)
    ap.add_argument("--max-clade", type=int, default=12)
    ap.add_argument("--confounding", type=float, default=1.0)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--save-z", default="", help="prefix for per-replicate z/R arrays")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    gene_rows, summary = [], []
    for rep in range(args.reps):
        t0 = time.time()
        tree = yule_tree(args.tips, rng)
        trait = pick_convergent_clades(tree, args.origins, (args.min_clade, args.max_clade), rng)
        fg = foreground_branches(tree, trait)
        origin = origin_index(tree, trait)
        keep = tree.parent[: tree.n_branches] != tree.root
        jobs, truth, kinds = make_jobs(tree, fg, args.genes, rng, frac=args.frac,
                                       confounding=args.confounding, omega_mult=3.0,
                                       rate_mult=1.5, height=0.8)
        nt, aa, cod = estimate(jobs, args.workers)
        X = layers_from_lengths(nt, aa, cod)[:, :, keep]
        fk, ok = fg[keep], origin[keep]

        z = np.zeros((args.genes, len(LAYERS)))
        resid = np.zeros((args.genes, len(LAYERS), int(keep.sum())))
        p_looo = []
        for l in range(len(LAYERS)):
            S = learn_covariance(X[:, l])
            z[:, l] = gls_z(X[:, l], S, fk[None])[:, 0]
            resid[:, l] = gls_residuals(X[:, l], S, fk)
            p_looo.append(leave_one_origin_out_p(X[:, l], S, ok))
        detected = bh(acat(np.column_stack(p_looo)), 0.1)
        R = layer_correlation_from(resid)
        if args.save_z:
            np.savez_compressed(f"{args.save_z}_rep{rep}.npz", z=z, R=R, kinds=kinds.astype(str),
                                truth=truth, detected=detected)
        cfg = fit_configurations(z, R)

        expected = np.array([EXPECTED.get(k, (0, 0, 0)) for k in kinds], dtype=bool)
        tp = detected & truth
        layer_auc = [auc(cfg.layer_post[tp, l], expected[tp, l]) for l in range(len(LAYERS))]
        map_cfg = cfg.configs[cfg.map_config]
        exact = (map_cfg[tp] == expected[tp]).all(1).mean() if tp.any() else float("nan")
        naive_cfg = st.norm.sf(z) < 0.05                         # per-layer nominal rule
        exact_naive = (naive_cfg[tp] == expected[tp]).all(1).mean() if tp.any() else float("nan")
        weights = {"".join("1" if b else "0" for b in c): round(float(w), 4)
                   for c, w in zip(cfg.configs, cfg.config_weights)}
        summary.append(dict(rep=rep, convergent=int(truth.sum()), detected=int(detected.sum()),
                            true_detected=int(tp.sum()), false_detected=int((detected & ~truth).sum()),
                            auc_rate_nucleotide=layer_auc[0], auc_rate_protein=layer_auc[1],
                            auc_selection_proxy=layer_auc[2], map_exact=exact,
                            per_layer_rule_exact=exact_naive, pi0=weights["000"],
                            config_weights=json.dumps(weights)))
        for g in np.nonzero(detected | truth)[0]:
            gene_rows.append(dict(rep=rep, gene=int(g), kind=kinds[g], detected=bool(detected[g]),
                                  **{f"z_{n}": round(float(z[g, i]), 3) for i, n in enumerate(LAYERS)},
                                  **{f"post_{n}": round(float(cfg.layer_post[g, i]), 3)
                                     for i, n in enumerate(LAYERS)},
                                  map_config="".join("1" if b else "0" for b in map_cfg[g])))
        print(f"rep={rep}: {time.time() - t0:.0f}s  " + json.dumps(summary[-1]), flush=True)

    pd.DataFrame(gene_rows).to_csv(args.out, index=False)
    pd.DataFrame(summary).to_csv(args.out.replace(".csv", "_summary.csv"), index=False)


if __name__ == "__main__":
    main()
