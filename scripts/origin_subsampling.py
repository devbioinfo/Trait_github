"""How many independent origins does the gene-loss test need? (sample-size question of the
comparative method; Cornwallis & Griffin 2024, section 6.1)

For the subterranean trait (6 origins), every subset of 3, 4 and 5 origins is analysed: species of
the dropped origins are set to missing (neither trait nor background), and the calibrated loss
test (leave-one-origin-out GLS, simulated null with species x gene error) is rerun. Reported:
calls at q = 0.1 and how many of the calls made with all 6 origins are recovered. The loss model
(rates, fragilities, species error rates) is fitted once on the full data.

Usage: python -I scripts/origin_subsampling.py [--sims 50] [--workers 4]
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import convergence_toga as C  # noqa: E402
from extra_analyses import setup, trait_vec  # noqa: E402
from mlconv.gls import learn_covariance_missing  # noqa: E402
from mlconv.loss import fit_loss_rates, gene_false_loss  # noqa: E402
from mlconv.stats import bh  # noqa: E402
from mlconv.tree import origin_index, origin_nodes  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=50)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "results" / "extra" / "origin_subsampling.csv"))
    args = ap.parse_args()
    out = Path(args.out)
    done = pd.read_csv(out) if out.exists() else pd.DataFrame()

    tree, names, lineage, symbols, Yall = setup()
    trait = trait_vec("hypoxia_subterranean", names, lineage)
    origins = origin_nodes(tree, trait)
    D = tree.tip_sets()
    full = pd.read_csv(ROOT / "results" / "hypoxia_subterranean_toga" / "hypoxia_subterranean_genes.csv")
    full_calls = set(full[full.call_loss == True].symbol)  # noqa: E712
    e = C.false_loss_rates(Yall)
    ok0 = C.testable(Yall)
    lam0, kap0 = fit_loss_rates(tree, Yall[ok0], e)
    lam_all = pd.Series(lam0, index=symbols[ok0])
    kap_all = pd.Series(kap0, index=symbols[ok0])
    labels = ["+".join(names[t].split("_")[0] for t in np.nonzero(D[o])[0][:2]) for o in origins]

    for k in (6, 5, 4, 3):
        for subset in itertools.combinations(range(len(origins)), k):
            key = ";".join(labels[i] for i in subset)
            if len(done) and (done.subset == key).any():
                continue
            keep_t = np.zeros(tree.n_tips, bool)
            for i in subset:
                keep_t |= D[origins[i]]
            drop = trait & ~keep_t
            Y = Yall.copy()
            Y[:, drop] = np.nan
            ok = C.testable(Y) & np.isin(symbols, lam_all.index)
            Y, sym = Y[ok], symbols[ok]
            lam, kap = lam_all[sym].to_numpy(), kap_all[sym].to_numpy()
            Sig = learn_covariance_missing(Y)
            o = origin_index(tree, keep_t)[: tree.n_tips]
            strata = np.searchsorted(np.quantile(lam, np.linspace(0, 1, 11)[1:-1]), lam, side="right")
            with Pool(args.workers, initializer=C._sim_init,
                      initargs=(dict(tree=tree, lam=lam, e=gene_false_loss(e, kap), missing=np.isnan(Y)),)) as pool:
                p = C.sim_p(C.loss_stat(Y, Sig, o), C.sim_null(pool, o, args.sims), strata)
            called = set(sym[bh(p, 0.1)])
            row = dict(n_origins=k, subset=key, calls=len(called), recovered=len(called & full_calls),
                       full_calls=len(full_calls), new=len(called - full_calls), genes=";".join(sorted(called)))
            done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
            done.to_csv(out, index=False)
            print(f"{k} origins [{key}]: {row['calls']} calls, {row['recovered']}/{len(full_calls)} recovered", flush=True)


if __name__ == "__main__":
    main()
