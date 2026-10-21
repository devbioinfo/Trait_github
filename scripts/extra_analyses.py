"""Follow-up analyses suggested by the phylogenetic comparative methods literature
(Cornwallis & Griffin 2024, Annu. Rev. Ecol. Evol. Syst. 55:181-204).

1. timing   When was each convergent gene lost relative to each trait origin? Under the
            irreversible-loss (Dollo) model, a node is lost when every observed species below it
            is lost. Per gene and origin: lost before the origin (shared with the sister
            lineage), on the origin's stem branch, later within the origin, or intact.
2. joint    Which of two correlated traits explains a loss? GLS on the species loss matrix with
            both trait indicators in one model (subterranean + high altitude; fully marine +
            semi-aquatic), giving partial effects (difference in loss probability) per gene.
3. quality  Is the fitted species false-loss rate related to assembly contiguity (contig N50)?
4. adequacy Do data simulated from the fitted loss model reproduce the observed number of losses
            per gene and per species?

Outputs in results/extra/. Usage: python -I scripts/extra_analyses.py
"""
from __future__ import annotations

import os
import sys
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
from mlconv.loss import fit_loss_rates, gene_false_loss, simulate_losses  # noqa: E402
from mlconv.tree import from_newick, origin_nodes  # noqa: E402

RES = ROOT / "results"
OUT = RES / "extra"
OUT.mkdir(exist_ok=True)


def setup():
    tree, names = from_newick((RES / "toga" / "master_paper.nwk").read_text())
    cov = pd.read_csv(RES / "toga" / "coverage_paper.tsv", sep="\t").drop_duplicates("species").set_index("species")
    lineage = cov.lineage.reindex(names).fillna("")
    m = pd.read_csv(RES / "toga" / "loss_status_paper.tsv.gz", sep="\t", index_col=0).reindex(columns=names).fillna("M")
    Yall = np.where(m.isin(["L", "UL"]), 1.0, np.where(m.isin(["I", "PI"]), 0.0, np.nan))
    return tree, names, lineage, m.index.to_numpy(), Yall


def trait_vec(key, names, lineage):
    return np.array([C.TRAITS[key][0](s, l) for s, l in zip(names, lineage)])


# ---------------------------------------------------------------------------------------
# 1. timing of convergent losses relative to trait origins
# ---------------------------------------------------------------------------------------

CATEGORIES = ["before origin", "at origin", "within origin", "intact", "no data"]


def timing(tree, names, lineage, symbols, Yall, key, genes):
    trait = trait_vec(key, names, lineage)
    D = tree.tip_sets()
    idx = {s: i for i, s in enumerate(symbols)}
    rows = []
    for o in origin_nodes(tree, trait):
        tips = np.nonzero(D[o])[0]
        p = tree.parent[o]
        sister = np.nonzero(D[p] & ~D[o])[0]
        label = "+".join(names[t].split("_")[0] for t in tips[:2]) + ("+..." if len(tips) > 2 else "")
        for g in genes:
            y = Yall[idx[g]]
            obs = tips[~np.isnan(y[tips])]
            if len(obs) == 0:
                cat = "no data"
            elif (y[obs] == 1).all():
                sobs = sister[~np.isnan(y[sister])]
                cat = "before origin" if len(sobs) and (y[sobs] == 1).all() else "at origin"
            elif (y[obs] == 1).any():
                cat = "within origin"
            else:
                cat = "intact"
            rows.append(dict(trait=key, gene=g, origin=label, origin_species=len(tips),
                             observed=len(obs), lost=int((y[obs] == 1).sum()), category=cat))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------
# 2. joint models: partial effects of two correlated traits on gene loss
# ---------------------------------------------------------------------------------------

def gls_multi(Y, Sig, X):
    """Per-gene GLS of Y (genes, species; NaN missing) on design X (species, k) with an intercept.
    Returns beta (genes, k) and t (genes, k); genes are fitted on their observed species."""
    G = len(Y)
    k = X.shape[1]
    beta = np.full((G, k), np.nan)
    tval = np.full((G, k), np.nan)
    obs = ~np.isnan(Y)
    patterns, inv = np.unique(obs, axis=0, return_inverse=True)
    for j, pat in enumerate(patterns):
        rows = np.nonzero(inv.ravel() == j)[0]
        Xd = np.column_stack([np.ones(pat.sum()), X[pat]])
        if np.linalg.matrix_rank(Xd) < Xd.shape[1] or pat.sum() < Xd.shape[1] + 5:
            continue
        L = np.linalg.cholesky(Sig[np.ix_(pat, pat)])
        Xw = np.linalg.solve(L, Xd)
        Yw = np.linalg.solve(L, Y[np.ix_(rows, pat)].T).T
        XtX_inv = np.linalg.inv(Xw.T @ Xw)
        b = Yw @ Xw @ XtX_inv
        res = Yw - b @ Xw.T
        s2 = (res ** 2).sum(1) / (pat.sum() - Xd.shape[1])
        se = np.sqrt(np.outer(s2, np.diag(XtX_inv)))
        beta[rows] = b[:, 1:]
        tval[rows] = (b / np.where(se > 0, se, np.nan))[:, 1:]
    return beta, tval


def joint(names, lineage, symbols, Yall, a_key, b_fn, labels):
    a = trait_vec(a_key, names, lineage).astype(float)
    b = np.array([b_fn(s, l) for s, l in zip(names, lineage)], dtype=float)
    ok = C.testable(Yall)
    Y, sym = Yall[ok], symbols[ok]
    Sig = learn_covariance_missing(Y)
    beta, t = gls_multi(Y, Sig, np.column_stack([a, b]))
    return pd.DataFrame({"gene": sym, f"beta_{labels[0]}": beta[:, 0], f"beta_{labels[1]}": beta[:, 1],
                         f"t_{labels[0]}": t[:, 0], f"t_{labels[1]}": t[:, 1],
                         f"lost_{labels[0]}": ((Y == 1) & (a[None] == 1)).sum(1),
                         f"lost_{labels[1]}": ((Y == 1) & (b[None] == 1)).sum(1),
                         "lost_other": ((Y == 1) & (a[None] == 0) & (b[None] == 0)).sum(1)})


# ---------------------------------------------------------------------------------------
# 3-4. data quality and model adequacy
# ---------------------------------------------------------------------------------------

def quality_and_adequacy(tree, names, Yall, n_rep=20):
    from mlconv.loss import false_loss_rates
    e = false_loss_rates(Yall)
    ov = pd.read_csv(RES / "toga" / "assemblies_paper.tsv", sep="\t").set_index("species")
    q = pd.DataFrame({"species": names, "false_loss_rate": e,
                      "contig_n50": ov["contig N50 (bp)"].reindex(names).to_numpy(),
                      "scaffold_n50": ov["scaffold N50 (bp)"].reindex(names).to_numpy()})
    ok = C.testable(Yall)
    Y = Yall[ok]
    lam, kappa = fit_loss_rates(tree, Y, e)
    rng = np.random.default_rng(5)
    per_gene, per_species = [], []
    for _ in range(n_rep):
        S = simulate_losses(tree, lam, gene_false_loss(e, kappa), np.isnan(Y), rng)
        S = S[C.testable(S)]
        per_gene.append(np.nansum(S == 1, 1))
        per_species.append(np.nanmean(S, 0))
    obs_gene = np.nansum(Y == 1, 1)
    bins = np.array([2, 3, 4, 5, 7, 10, 15, 25, 40, 60, 116])
    obs_h = np.histogram(obs_gene, bins)[0] / len(obs_gene)
    sim_h = np.array([np.histogram(g, bins)[0] / len(g) for g in per_gene])
    adequacy_gene = pd.DataFrame({"bin_low": bins[:-1], "bin_high": bins[1:] - 1, "observed": obs_h,
                                  "sim_mean": sim_h.mean(0), "sim_low": sim_h.min(0), "sim_high": sim_h.max(0)})
    sp = np.array(per_species)
    adequacy_species = pd.DataFrame({"species": names, "observed": np.nanmean(Y, 0),
                                     "sim_mean": sp.mean(0), "sim_low": sp.min(0), "sim_high": sp.max(0)})
    model = pd.DataFrame({"gene_index": np.arange(len(lam)), "loss_rate": lam, "fragility": kappa})
    return q, adequacy_gene, adequacy_species, model


def main():
    tree, names, lineage, symbols, Yall = setup()
    genes = {}
    for k in ("marine", "hypoxia_subterranean"):
        d = pd.read_csv(RES / f"{k}_toga" / f"{k}_genes.csv")
        genes[k] = d[d.call_loss == True].sort_values("p_loss").symbol.tolist()  # noqa: E712
    tm = pd.concat([timing(tree, names, lineage, symbols, Yall, k, g) for k, g in genes.items()])
    tm.to_csv(OUT / "loss_timing.csv", index=False)
    print(tm.groupby(["trait", "category"]).size().unstack(fill_value=0))

    hyp = joint(names, lineage, symbols, Yall, "hypoxia_subterranean", C._is(C.ALTITUDE), ("subterranean", "altitude"))
    hyp.to_csv(OUT / "joint_subterranean_altitude.csv", index=False)
    semi = lambda sp, l: ("Lutrinae" in l or sp.startswith(("Castor_", "Ondatra_")) or "Hippopotamidae" in l)  # noqa: E731
    mar = joint(names, lineage, symbols, Yall, "marine", semi, ("marine", "semiaquatic"))
    mar.to_csv(OUT / "joint_marine_semiaquatic.csv", index=False)
    for df, lab, g in ((hyp, "subterranean", genes["hypoxia_subterranean"]), (mar, "marine", genes["marine"])):
        sub = df[df.gene.isin(g)]
        cols = [c for c in df.columns if c.startswith("beta_")]
        print(lab, "called genes: median partial effects", sub[cols].median().round(3).to_dict())

    q, ag, asp, model = quality_and_adequacy(tree, names, Yall)
    q.to_csv(OUT / "species_quality.csv", index=False)
    ag.to_csv(OUT / "adequacy_losses_per_gene.csv", index=False)
    asp.to_csv(OUT / "adequacy_losses_per_species.csv", index=False)
    model.to_csv(OUT / "loss_model_parameters.csv", index=False)
    r = st.spearmanr(np.log10(q.contig_n50), q.false_loss_rate, nan_policy="omit")
    print(f"false-loss rate vs log10 contig N50: Spearman rho = {r.statistic:.2f}, p = {r.pvalue:.2g}")


if __name__ == "__main__":
    main()
