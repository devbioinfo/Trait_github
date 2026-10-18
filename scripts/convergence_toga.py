"""Genome-wide convergence test on Zoonomia/TOGA genes with the final pipeline.

Input: results of scripts/toga_branch_lengths.py (master.nwk, branch_lengths.jsonl) and
TOGA's coverage table (lineage of each species).

Traits (--trait), defined from the taxonomic lineage before any test:
  echolocation  laryngeal echolocation: bats other than Pteropodidae, plus toothed whales
                (Odontoceti); three origins (Rhinolophoidea, Yangochiroptera, Odontoceti)
  marine        fully marine mammals: Cetacea, Pinnipedia (Otariidae, Phocidae, Odobenidae)
                and Sirenia; semi-aquatic species (otters, hippopotamus, beaver, muskrat,
                polar bear) are kept out of the control trait sets
  aquatic       marine plus semi-aquatic rodents (beaver, muskrat); hippopotamus, otters and
                polar bear are kept out of the control sets. Otters are not trait species
                because, without other musteloids in the species set, the giant otter is
                sister to the pinnipeds and the two would merge into one origin with a
                (wrongly) aquatic common ancestor
Each trait has a list of genes of interest (literature candidates), reported separately.

Pipeline (as validated on simulations):
  layers  log nucleotide rate, log protein rate, selection proxy (difference); pseudo-count of
          one substitution per alignment; mean-centred per branch; root edges excluded
  test    per layer GLS with a genome-learned covariance (pairwise-complete, missing branches
          allowed), leave-one-origin-out intersection-union test, ACAT over layers, BH FDR
  loss    optional (--loss): convergent gene loss from TOGA's ortholog classification
          (scripts/toga_loss_matrix.py). Each species is coded lost (1; --loss-states,
          default L and UL), intact (0; I and PI) or missing (M). The same GLS is fitted
          across species instead of branches, with a species x species covariance learned
          from the loss patterns of all genes (it absorbs phylogeny and assembly quality: a
          poor assembly with many spurious losses is down-weighted), and the same
          leave-one-origin-out test, so a loss counts as convergent only if it is shared by
          at least two independent origins. Genes need >= 2 lost and >= 2 intact species
          (a filter blind to the trait). The GLS p-value is not calibrated for 0/1 data
          (clade-inherited losses; 60% of marine control sets had calls), so the loss
          p-value comes from a null distribution of the same statistic:
            sim       (default) data simulated from a fitted irreversible-loss model with a
                      loss rate per gene and a false-loss rate per species (mlconv/loss.py);
                      genes are pooled within bins of fitted loss rate, and the far tail is
                      extrapolated with a generalized Pareto fit
            controls  the statistic on --loss-null matched control trait sets, pooled over
                      genes with a similar number of losses (limited and conservative when
                      the tree offers few distinct control sets)
          Rate and loss evidence are combined by ACAT.
  signatures  mash-style model with learned directions on pooled per-layer z-scores
  calibration the same test on control trait sets (clades without the trait, matched to the
          origins in size and age); every call for a control set is false

Usage: python -I scripts/convergence_toga.py --trait marine --bl /path/to/bl --coverage coverage.tsv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# many small (~200 x 200) factorisations: multithreaded BLAS only adds overhead, and is ~200x
# slower when other jobs share the CPU
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mlconv.loss import false_loss_rates, fit_loss_rates, simulate_losses, tail_p  # noqa: E402
from mlconv.gls import gls_z_missing, learn_covariance_missing, leave_one_origin_out_p_missing  # noqa: E402
from mlconv.signatures import fit_signatures, pattern_active  # noqa: E402
from mlconv.stats import acat, bh  # noqa: E402
from mlconv.tree import (foreground_branches, from_newick, matched_clade_permute, origin_index,  # noqa: E402
                         origin_nodes)

LAYERS = ("rate_nucleotide", "rate_protein", "selection_proxy")
MARINE = ("Cetacea", "Otariidae", "Phocidae", "Odobenidae", "Sirenia")
SEMI_AQUATIC = ("Lutrinae", "Castor_", "Ondatra_")
AQUATIC_RODENTS = ("Castor_", "Ondatra_")
TRAITS = {  # name: (trait test on (species, lineage), species kept out of control sets, genes of interest)
    "echolocation": (
        lambda sp, l: ("Chiroptera" in l and "Pteropodidae" not in l) or "Odontoceti" in l,
        lambda sp, l: False,
        ["SLC26A5", "TMC1", "PJVK", "CDH23", "PCDH15", "OTOF", "KCNQ4", "TECTA", "STRC", "CABP2",
         "OTOA", "LOXHD1", "ESPN", "GJB2", "MYO15A", "USH1C", "TMPRSS3", "SLC17A8"]),
    "marine": (
        lambda sp, l: any(k in l for k in MARINE),
        lambda sp, l: any(k in l or sp.startswith(k) for k in SEMI_AQUATIC)
        or "Hippopotamidae" in l or sp == "Ursus_maritimus",
        # marine gene losses / shifts reported in cetaceans, pinnipeds and sirenians
        ["PON1", "MMP12", "KLK8", "MTNR1A", "MTNR1B", "AANAT", "ASMT", "F12", "SEC14L3", "DSG4",
         "GSDMA", "TGM3", "SLC6A18", "POLM", "NAT16", "TAS1R1", "TAS1R2", "TAS1R3", "OMP"]),
    "aquatic": (
        lambda sp, l: any(k in l for k in MARINE) or any(sp.startswith(k) for k in AQUATIC_RODENTS),
        lambda sp, l: "Hippopotamidae" in l or "Lutrinae" in l or sp == "Ursus_maritimus",
        ["PON1", "MMP12", "KLK8", "MTNR1A", "MTNR1B", "AANAT", "ASMT", "F12", "SEC14L3", "DSG4",
         "GSDMA", "TGM3", "SLC6A18", "POLM", "NAT16", "TAS1R1", "TAS1R2", "TAS1R3", "OMP"]),
}


def load(bl_dir: Path):
    tree, names = from_newick((bl_dir / "master.nwk").read_text())
    genes, nt, aa, codons = [], [], [], []
    for ln in (bl_dir / "branch_lengths.jsonl").read_text().splitlines():
        r = json.loads(ln)
        if r.get("status") != "ok":
            continue
        genes.append(r["gene"])
        nt.append([np.nan if v is None else v for v in r["nt"]])
        aa.append([np.nan if v is None else v for v in r["aa"]])
        codons.append(r["codons"])
    return tree, names, genes, np.array(nt), np.array(aa), np.array(codons, float)


def layers(nt, aa, codons, keep):
    ln_nt = np.log(nt + 1.0 / (3 * codons)[:, None])
    ln_aa = np.log(aa + 1.0 / codons[:, None])
    X = np.stack([ln_nt, ln_aa, ln_aa - ln_nt], axis=1)[:, :, keep]
    return X - np.nanmean(X, axis=0, keepdims=True)


def final_test(X, Sig, origin):
    p = np.column_stack([leave_one_origin_out_p_missing(X[:, l], Sig[l], origin) for l in range(X.shape[1])])
    return acat(np.clip(p, 1e-300, 1 - 1e-12))


def testable(Y):
    return ((Y == 1).sum(1) >= 2) & ((Y == 0).sum(1) >= 2)


def loss_data(path, names, lost_states):
    """(symbols, Y, e): Y (genes, species) 1 lost, 0 intact, NaN missing, testable genes only;
    e the per-species false-loss rate estimated from all genes."""
    m = pd.read_csv(path, sep="\t", index_col=0).reindex(columns=names).fillna("M")
    Y = np.where(m.isin(lost_states), 1.0, np.where(m.isin(["I", "PI"]), 0.0, np.nan))
    ok = testable(Y)
    return m.index[ok].to_numpy(), Y[ok], false_loss_rates(Y)


SIM: dict = {}


def _sim_init(ctx):
    SIM.update(ctx)
    os.environ["OMP_NUM_THREADS"] = "1"


def _sim_stat(task):
    """Statistic on one simulated replicate for one trait configuration (NaN = not testable)."""
    seed, tip_origin = task
    Ys = simulate_losses(SIM["tree"], SIM["lam"], SIM["e"], SIM["missing"], np.random.default_rng(seed))
    ok = testable(Ys)
    out = np.full(len(Ys), np.nan)
    out[ok] = loss_stat(Ys[ok], learn_covariance_missing(Ys[ok]), tip_origin)
    return out


def sim_null(pool, tip_origin, n_sims, seed=0):
    """(n_sims, genes) null statistics for one trait configuration."""
    return np.array(pool.map(_sim_stat, [(seed + k, tip_origin) for k in range(n_sims)]))


def sim_p(stat, null, strata):
    """Tail p-value within each stratum of genes (pooling genes and replicates)."""
    p = np.ones(len(stat))
    for k in np.unique(strata):
        g = strata == k
        p[g] = tail_p(stat[g], null[:, g].ravel())
    return p


LOSS_BINS = np.array([2, 3, 4, 5, 7, 10, 15, 25, 40, 60])


def loss_strata(Y):
    """Stratum of each gene by its number of lost species (the null spread depends on it)."""
    return np.searchsorted(LOSS_BINS, (Y == 1).sum(1), side="right") - 1


def loss_stat(Y, Sig, tip_origin):
    """-log of the leave-one-origin-out GLS p-value (0 when untestable)."""
    return -np.log(np.clip(leave_one_origin_out_p_missing(Y, Sig, tip_origin), 1e-300, 1))


def empirical_p(stat, null, strata):
    """P(null statistic >= observed) within each stratum, pooling genes and null sets.

    null: (K, G) statistics on K control trait sets; p = (1 + #exceed) / (1 + #null).
    """
    p = np.ones(len(stat))
    for k in np.unique(strata):
        g = strata == k
        pool = np.sort(null[:, g].ravel())
        exceed = len(pool) - np.searchsorted(pool, stat[g], side="left")
        p[g] = (1 + exceed) / (1 + len(pool))
    return p


def combine(p_rate, p_loss):
    """ACAT of the rate and loss p-values where both exist, else whichever exists.

    Inputs are capped at 0.99: a Cauchy term for p -> 1 tends to -infinity and would cancel
    any evidence in the other layer (e.g. a gene lost in the trait lineages has no rate data
    there, so its leave-one-origin-out rate p-value is 1).
    """
    P = np.column_stack([p_rate, p_loss])
    out = np.full(len(P), np.nan)
    both = ~np.isnan(P).any(1)
    out[both] = acat(np.clip(P[both], 1e-300, 0.99))
    one = ~both & ~np.isnan(P).all(1)
    out[one] = np.nanmin(P[one], 1)
    return out


def bh_nan(p, q):
    """BH over the genes with a p-value; NaN never rejected."""
    rej = np.zeros(len(p), dtype=bool)
    ok = ~np.isnan(p)
    rej[ok] = bh(p[ok], q)
    return rej


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trait", required=True, choices=sorted(TRAITS))
    ap.add_argument("--bl", required=True, help="directory with master.nwk and branch_lengths.jsonl")
    ap.add_argument("--species", default=str(ROOT / "results" / "toga" / "species.tsv"))
    ap.add_argument("--coverage", required=True, help="TOGA coverage table with lineages")
    ap.add_argument("--out", help="default: results/<trait>_toga")
    ap.add_argument("--controls", type=int, default=20)
    ap.add_argument("--q", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--loss", help="gene-loss status matrix from scripts/toga_loss_matrix.py")
    ap.add_argument("--loss-states", default="L,UL", help="TOGA classes counted as lost")
    ap.add_argument("--loss-null-mode", choices=("sim", "controls"), default="sim")
    ap.add_argument("--loss-null", type=int, default=200,
                    help="controls mode: control trait sets used to calibrate the loss layer "
                         "(disjoint from --controls)")
    ap.add_argument("--loss-sims", type=int, default=100, help="sim mode: simulated replicates")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true",
                    help="keep control sets already in <out>/control_calls.csv (same seed) and run the rest")
    args = ap.parse_args()
    out = Path(args.out or ROOT / "results" / f"{args.trait}_toga")
    is_trait, is_excluded, genes_of_interest = TRAITS[args.trait]
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    tree, names, genes, nt, aa, codons = load(Path(args.bl))
    symbols = [g.split(".", 1)[1] for g in genes]
    keep = tree.parent[: tree.n_branches] != tree.root
    X = layers(nt, aa, codons, keep)
    print(f"{len(genes)} genes, {tree.n_tips} species, {int(keep.sum())} branches; "
          f"missing branch values {np.isnan(X[:, 0]).mean():.1%}", flush=True)

    cov = pd.read_csv(args.coverage, sep="\t").drop_duplicates("species").set_index("species")
    lineage = cov.lineage.reindex(names).fillna("")
    trait = np.array([is_trait(sp, l) for sp, l in zip(names, lineage)])
    excluded = np.array([is_excluded(sp, l) for sp, l in zip(names, lineage)]) & ~trait
    fg = foreground_branches(tree, trait)
    origin = origin_index(tree, trait)[keep]
    origins = origin_nodes(tree, trait)
    print(f"{args.trait}: {trait.sum()} species, {len(origins)} origins, {int(fg[keep].sum())} "
          f"foreground branches; kept out of controls: {', '.join(np.array(names)[excluded]) or 'none'}",
          flush=True)
    (out / "trait_species.txt").write_text("\n".join(n for n, t in zip(names, trait) if t) + "\n")

    # matched control trait sets: the first --controls check calibration, the rest calibrate
    # the loss layer
    D = tree.tip_sets()
    cand = np.unique(matched_clade_permute(tree, origins, 10 * (args.controls + args.loss_null), rng,
                                           exclude=trait | excluded, size_tol=1), axis=0)
    ctrl_tips = []
    for c in cand[rng.permutation(len(cand))]:
        tips = np.array([D[np.nonzero(c)[0], t].any() for t in range(tree.n_tips)])
        if len(origin_nodes(tree, tips)) == len(origins):
            ctrl_tips.append(tips)
    eval_tips, null_tips = ctrl_tips[: args.controls], ctrl_tips[args.controls: args.controls + args.loss_null]
    print(f"control trait sets: {len(eval_tips)} for checking"
          + (f", {len(null_tips)} for loss calibration" if args.loss_null_mode == "controls" else ""),
          flush=True)

    t0 = time.time()
    Sig = [learn_covariance_missing(X[:, l]) for l in range(3)]
    p_rate = final_test(X, Sig, origin)
    z = np.column_stack([gls_z_missing(X[:, l], Sig[l], fg[keep].astype(float)) for l in range(3)])
    res = pd.DataFrame({"gene": genes, "symbol": symbols, "p_rate": p_rate,
                        **{f"z_{n}": z[:, i] for i, n in enumerate(LAYERS)}})
    res = res.drop_duplicates("symbol")
    tip_origin = origin_index(tree, trait)[: tree.n_tips]
    if args.loss:
        lost_states = args.loss_states.split(",")
        loss_sym, Yl, e_false = loss_data(args.loss, names, lost_states)
        Sig_l = learn_covariance_missing(Yl)
        p_loss_gls = np.exp(-loss_stat(Yl, Sig_l, tip_origin))
        if args.loss_null_mode == "sim":
            lam = fit_loss_rates(tree, Yl, e_false)
            strata = np.searchsorted(np.quantile(lam, np.linspace(0, 1, 11)[1:-1]), lam, side="right")
            from multiprocessing import Pool
            sim_pool = Pool(args.workers, initializer=_sim_init,
                            initargs=(dict(tree=tree, lam=lam, e=e_false, missing=np.isnan(Yl)),))

            def loss_p(tip_orig):
                return sim_p(loss_stat(Yl, Sig_l, tip_orig), sim_null(sim_pool, tip_orig, args.loss_sims), strata)
        else:
            strata = loss_strata(Yl)
            null = np.array([loss_stat(Yl, Sig_l, origin_index(tree, tips)[: tree.n_tips]) for tips in null_tips])

            def loss_p(tip_orig):
                return empirical_p(loss_stat(Yl, Sig_l, tip_orig), null, strata)
        p_loss = loss_p(tip_origin)
        z_loss = gls_z_missing(Yl, Sig_l, trait.astype(float))
        lost_in = [";".join(np.array(names)[(row == 1) & trait]) for row in Yl]
        loss = pd.DataFrame({"symbol": loss_sym, "p_loss": p_loss, "p_loss_gls": p_loss_gls, "z_loss": z_loss,
                             "trait_species_lost": lost_in,
                             "n_lost_trait": ((Yl == 1) & trait).sum(1), "n_lost_other": ((Yl == 1) & ~trait).sum(1)})
        res = res.merge(loss, on="symbol", how="outer")
        print(f"loss layer: {len(loss_sym)} testable genes (lost = {'+'.join(lost_states)})", flush=True)
    else:
        res["p_loss"] = np.nan
    res["p_final"] = combine(res.p_rate.to_numpy(), res.p_loss.to_numpy())
    for col in ("p_rate", "p_loss", "p_final"):
        rej_c = bh_nan(res[col].to_numpy(), args.q)
        res[col.replace("p_", "call_")] = rej_c
        print(f"{args.trait}: {rej_c.sum()} genes at q = {args.q} [{col[2:]}]", flush=True)
    print(f"({time.time() - t0:.0f}s)", flush=True)
    rej = res.call_final.to_numpy()

    # signatures: rate layers (+ loss) z-scores, genome-wide null correlation from all genes
    lay = list(LAYERS) + (["loss"] if args.loss else [])
    zc = np.nan_to_num(res[[f"z_{n}" for n in lay]].to_numpy())
    R = np.corrcoef(zc.T)
    sig = fit_signatures(zc, np.broadcast_to(R, (len(zc), len(lay), len(lay))).copy(), lay)
    active = pattern_active(sig)
    res["signature"] = ["+".join(n for n, a in zip(lay, row) if a) for row in active]
    res.loc[~rej, "signature"] = ""
    res = res.sort_values("p_final")
    res.to_csv(out / f"{args.trait}_genes.csv", index=False)
    pd.Series({"_".join(n.split(":")[0:1]) + str(i): float(w) for i, ((n, _), w) in
               enumerate(zip(sig.patterns, sig.pattern_weights))}).to_csv(out / "signature_patterns.csv")
    show = ["symbol", "p_rate", "p_loss", "p_final"] + [f"z_{n}" for n in lay] + \
           (["n_lost_trait", "n_lost_other"] if args.loss else []) + ["signature"]
    fmt = dict(index=False, float_format=lambda v: f"{v:.3g}")
    print("\nGenes of interest:")
    print(res[res.symbol.isin(genes_of_interest)][show].to_string(**fmt))
    print("\nTop 30:")
    print(res[show].head(30).to_string(**fmt))

    # calibration check on matched control trait sets (never used to calibrate)
    rows = []
    if args.resume and (out / "control_calls.csv").exists():
        rows = pd.read_csv(out / "control_calls.csv").to_dict("records")
        print(f"resuming after {len(rows)} control sets", flush=True)
    for tips in eval_tips[len(rows):]:
        oc = origin_index(tree, tips)
        pr = pd.Series(final_test(X, Sig, oc[keep]), index=symbols)
        pr = pr[~pr.index.duplicated()]
        row = dict(control=len(rows), calls_rate=int(bh(pr.to_numpy(), args.q).sum()), min_p_rate=float(pr.min()))
        if args.loss:
            pl = pd.Series(loss_p(oc[: tree.n_tips]), index=loss_sym)
            u = pr.index.union(pl.index)
            pf = combine(pr.reindex(u).to_numpy(), pl.reindex(u).to_numpy())
            row.update(calls_loss=int(bh(pl.to_numpy(), args.q).sum()), min_p_loss=float(pl.min()),
                       calls_final=int(bh_nan(pf, args.q).sum()))
        else:
            row["calls_final"] = row["calls_rate"]
        row["species"] = ";".join(n for n, t in zip(names, tips) if t)
        rows.append(row)
        print(f"control {len(rows)}: " + ", ".join(f"{k} {v}" for k, v in row.items() if k.startswith("calls")),
              flush=True)
        pd.DataFrame(rows).to_csv(out / "control_calls.csv", index=False)
    cc = pd.DataFrame(rows)
    summary = dict(trait=args.trait, genes=len(genes), species=tree.n_tips, trait_species=int(trait.sum()),
                   origins=len(origins), loss_states=args.loss_states if args.loss else None,
                   loss_null=(args.loss_null_mode if args.loss else None),
                   control_sets=len(cc))
    for k in ("rate", "loss", "final"):
        if f"call_{k}" in res and (k != "loss" or args.loss):
            summary[f"calls_{k}"] = int(res[f"call_{k}"].sum())
        if f"calls_{k}" in cc:
            summary[f"control_mean_calls_{k}"] = float(cc[f"calls_{k}"].mean())
            summary[f"control_sets_with_any_call_{k}"] = float((cc[f"calls_{k}"] > 0).mean())
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
