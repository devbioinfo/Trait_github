"""Calibration and power of the gene-loss layer on simulated genomes.

Genomes are simulated on the real 115-species tree from the loss model fitted to the real
TOGA loss matrix: every gene keeps its fitted loss rate and fragility, every species its
false-loss rate, and every gene its observed missing-data pattern. A random trait with
3, 6 or 10 independent origins (clades of 1-4 species) is placed on the tree, and in 2% of
genes each origin loses the gene on its stem branch with probability `penetrance`
(0 = null genome). The simulated genome is then analysed exactly as real data: the model
is refitted, the testability filter applied, and loss p-values computed by

  simulation  the method of the paper (simulated null, fragility-aware; 100 replicates)
  gls         the same statistic with its parametric GLS p-value (no simulation null)
  fisher      one-sided Fisher exact test of lost vs intact in trait vs other species,
              ignoring phylogeny (a common naive approach)

with Benjamini-Hochberg at q = 0.1.

Usage: python -I scripts/loss_benchmark.py --bl BL_DIR --loss loss_status.tsv.gz --out results/loss_benchmark.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as st  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import convergence_toga as C  # noqa: E402
from mlconv.gls import learn_covariance_missing  # noqa: E402
from mlconv.loss import FALSE_INTACT, fit_loss_rates, gene_false_loss, simulate_losses  # noqa: E402
from mlconv.stats import bh  # noqa: E402
from mlconv.tree import origin_index, origin_nodes  # noqa: E402


def random_trait(tree, n_origins, rng, max_size=4):
    """Tip trait with exactly n_origins independent clades of 1..max_size species."""
    D = tree.tip_sets()
    sizes = D.sum(1)
    cand = [v for v in range(tree.root) if sizes[v] <= max_size]
    for _ in range(1000):
        rng.shuffle(cand)
        trait = np.zeros(tree.n_tips, dtype=bool)
        count = 0
        for v in cand:
            if (trait & D[v]).any():
                continue
            trial = trait | D[v]
            if len(origin_nodes(tree, trial)) == count + 1:      # not merged with a sister origin
                trait, count = trial, count + 1
                if count == n_origins:
                    return trait
    raise ValueError("could not place origins")


def plant_losses(tree, Y, trait, genes, penetrance, rng):
    """Lose `genes` on the stem branch of each origin with probability `penetrance`."""
    D = tree.tip_sets()
    Y = Y.copy()
    for o in origin_nodes(tree, trait):
        hit = genes[rng.random(len(genes)) < penetrance]
        tips = np.nonzero(D[o])[0]
        for t in tips:
            obs = ~np.isnan(Y[hit, t])
            vals = np.where(rng.random(len(hit)) < FALSE_INTACT, 0.0, 1.0)
            Y[hit[obs], t] = vals[obs]
    return Y


def fisher_p(Y, trait):
    """One-sided Fisher exact p for more losses among trait species (observed only)."""
    lost_t = np.nansum(Y[:, trait] == 1, 1)
    obs_t = (~np.isnan(Y[:, trait])).sum(1)
    lost = np.nansum(Y == 1, 1)
    obs = (~np.isnan(Y)).sum(1)
    return st.hypergeom.sf(lost_t - 1, obs, lost, obs_t)


def score(p, truth, q=0.1):
    p = np.where(np.isnan(p), 1.0, p)
    r = bh(p, q)
    tp = int((r & truth).sum())
    return dict(calls=int(r.sum()), tp=tp, fdp=(r.sum() - tp) / max(r.sum(), 1),
                power=tp / max(truth.sum(), 1) if truth.any() else np.nan)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bl", required=True)
    ap.add_argument("--loss", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--origins", default="3,6,10")
    ap.add_argument("--penetrance", default="0,0.5,0.75,1.0")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--frac", type=float, default=0.02)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    out = Path(args.out)
    done = pd.read_csv(out) if out.exists() else pd.DataFrame()

    tree, names, *_ = C.load(Path(args.bl))
    sym, Y_real, e = C.loss_data(args.loss, names, ["L", "UL"])
    lam0, kap0 = fit_loss_rates(tree, Y_real, e)
    missing = np.isnan(Y_real)
    print(f"{len(sym)} genes, {tree.n_tips} species; model fitted to the real loss matrix", flush=True)

    for n_orig in map(int, args.origins.split(",")):
        for pen in map(float, args.penetrance.split(",")):
            for rep in range(args.reps):
                key = (n_orig, pen, rep)
                if len(done) and ((done.origins == n_orig) & (done.penetrance == pen) & (done.rep == rep)).any():
                    continue
                t0 = time.time()
                rng = np.random.default_rng([args.seed, n_orig, int(pen * 100), rep])
                trait = random_trait(tree, n_orig, rng)
                Y = simulate_losses(tree, lam0, gene_false_loss(e, kap0), missing, rng)
                conv = rng.choice(len(Y), int(args.frac * len(Y)), replace=False) if pen > 0 else np.array([], int)
                Y = plant_losses(tree, Y, trait, conv, pen, rng)
                truth = np.zeros(len(Y), bool)
                truth[conv] = True
                ok = C.testable(Y)
                Y, truth = Y[ok], truth[ok]
                # analysed as real data: refit the model, learn the covariance, test
                e_s = e                                     # species rates from the full real matrix
                lam, kap = fit_loss_rates(tree, Y, e_s)
                Sig = learn_covariance_missing(Y)
                o = origin_index(tree, trait)[: tree.n_tips]
                stat = C.loss_stat(Y, Sig, o)
                strata = np.searchsorted(np.quantile(lam, np.linspace(0, 1, 11)[1:-1]), lam, side="right")
                with Pool(args.workers, initializer=C._sim_init,
                          initargs=(dict(tree=tree, lam=lam, e=gene_false_loss(e_s, kap), missing=np.isnan(Y)),)) as pool:
                    p_sim = C.sim_p(stat, C.sim_null(pool, o, args.sims, seed=10_000 + rep), strata)
                rows = []
                for method, p in (("simulation", p_sim), ("gls", np.exp(-stat)), ("fisher", fisher_p(Y, trait))):
                    rows.append(dict(origins=n_orig, penetrance=pen, rep=rep, method=method,
                                     trait_species=int(trait.sum()), genes=len(Y), convergent=int(truth.sum()),
                                     **score(p, truth)))
                done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
                done.to_csv(out, index=False)
                print(f"origins {n_orig} penetrance {pen} rep {rep} ({time.time() - t0:.0f}s): " +
                      "; ".join(f"{r['method']} calls {r['calls']} FDP {r['fdp']:.2f} power {r['power']:.2f}"
                                for r in rows), flush=True)


if __name__ == "__main__":
    main()
