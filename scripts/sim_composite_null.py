"""Error properties of the single-origin (intersection-union) check under the composite null.

For one trait configuration, G testable genes are simulated under the fitted irreversible-loss model
(background loss rate and fragility of each gene, its real missing-data pattern). In scenario k, the
gene is then truly lost in every observed species of k randomly chosen origins. The single-origin
check is applied exactly as in convergence_toga.py (background rates re-fitted from non-trait
species, covariance learned from the real data, n_sims replicates, Monte Carlo p-values).

k = 0 and k = 1 satisfy the composite null (at most one origin truly lost the gene): the share of
p-values <= alpha should not exceed alpha. k >= 2 gives power.

Output: <out>/composite_null_sim.csv (one row per gene and scenario) and a summary on screen.
Run on the Mac:  python3 scripts/sim_composite_null.py --trait hypoxia_subterranean --genes 300 --sims 2000
"""
import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import convergence_toga as ct  # noqa: E402
from mlconv.gls import learn_covariance_missing  # noqa: E402
from mlconv.loss import fit_loss_rates, gene_false_loss, simulate_losses  # noqa: E402
from mlconv.tree import from_newick, origin_index  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trait", default="hypoxia_subterranean", choices=list(ct.TRAITS))
    ap.add_argument("--tree", default=str(ROOT / "results/toga/master_paper.nwk"))
    ap.add_argument("--coverage", default=str(ROOT / "results/toga/coverage_paper.tsv"))
    ap.add_argument("--loss", default=str(ROOT / "results/toga/loss_status_paper.tsv.gz"))
    ap.add_argument("--genes", type=int, default=300, help="genes simulated per scenario")
    ap.add_argument("--sims", type=int, default=2000, help="replicates of the single-origin null per gene")
    ap.add_argument("--scenarios", default="0,1,2,3", help="numbers of origins with a true loss")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default=str(ROOT / "results/audit2"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    tree, names = from_newick(Path(args.tree).read_text())
    cov = pd.read_csv(args.coverage, sep="\t").drop_duplicates("species").set_index("species")
    lineage = cov.lineage.reindex(names).fillna("")
    is_trait = ct.TRAITS[args.trait][0]
    trait = np.array([is_trait(sp, l) for sp, l in zip(names, lineage)])
    tip_origin = origin_index(tree, trait)[: tree.n_tips]
    m = int(tip_origin.max()) + 1
    sym, Y, e_false = ct.loss_data(args.loss, names, ["L", "UL"])
    Sig = learn_covariance_missing(Y)
    Yb = Y.copy()
    Yb[:, trait] = np.nan
    lam, kappa = fit_loss_rates(tree, Yb, e_false)
    print(f"{args.trait}: {trait.sum()} species, {m} origins, {len(sym)} testable genes; "
          f"{args.genes} genes x {args.sims} replicates per scenario", flush=True)

    rows = []
    for k in [int(x) for x in args.scenarios.split(",")]:
        pick = rng.choice(len(sym), size=4 * args.genes, replace=False)
        miss = np.isnan(Y[pick])
        Ys = simulate_losses(tree, lam[pick], gene_false_loss(e_false, kappa[pick]), miss, rng)
        lost_origins = np.zeros((len(pick), m), bool)
        for g in range(len(pick)):
            for j in rng.choice(m, size=k, replace=False):
                lost_origins[g, j] = True
                cols = (tip_origin == j) & ~miss[g]
                Ys[g, cols] = 1.0
        ok = np.where(ct.testable(Ys))[0][: args.genes]
        Ys, miss, pick, lost_origins = Ys[ok], miss[ok], pick[ok], lost_origins[ok]
        # as in the real analysis: background rates re-fitted from the non-trait species of these data
        Yb = Ys.copy()
        Yb[:, trait] = np.nan
        lam_bg, kappa_bg = fit_loss_rates(tree, Yb, e_false)
        stat = ct.loss_stat(Ys, Sig, tip_origin)
        with Pool(args.workers, initializer=ct._sim_init,
                  initargs=(dict(tree=tree, missing=miss, Y=Ys, Sig=Sig),)) as pool:
            p, kij = ct.iut_p(pool, stat, np.arange(len(Ys)), lam_bg, gene_false_loss(e_false, kappa_bg),
                              tip_origin, args.sims, counts=True, seed=10**6 + 1000 * k)
        p_mc = ((kij + 1) / (args.sims + 1)).max(1)
        for g in range(len(Ys)):
            rows.append(dict(scenario_true_origins=k, gene=sym[pick[g]], stat=stat[g], p_iut_mc=p_mc[g],
                             p_iut_tail=p[g].max(), lost_origins=";".join(map(str, np.where(lost_origins[g])[0]))))
        d = pd.DataFrame([r for r in rows if r["scenario_true_origins"] == k])
        print(f"k = {k}: {len(d)} genes; share p <= 0.05: {np.mean(d.p_iut_mc <= 0.05):.3f}, "
              f"<= 0.01: {np.mean(d.p_iut_mc <= 0.01):.3f}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / f"composite_null_sim_{args.trait}.csv", index=False)
    summ = res.groupby("scenario_true_origins").agg(
        genes=("gene", "size"), share_p_le_0_05=("p_iut_mc", lambda v: np.mean(v <= 0.05)),
        share_p_le_0_01=("p_iut_mc", lambda v: np.mean(v <= 0.01)),
        median_p=("p_iut_mc", "median"))
    summ.to_csv(out / f"composite_null_sim_{args.trait}_summary.csv")
    print(summ.to_string())


if __name__ == "__main__":
    main()
