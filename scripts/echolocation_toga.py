"""Echolocation analysis on Zoonomia/TOGA genes with the final pipeline.

Input: results of scripts/toga_branch_lengths.py (master.nwk, branch_lengths.jsonl) and the
species table (results/toga/species.tsv) with lineages from TOGA's coverage table.

Trait: laryngeal echolocation = bats other than Pteropodidae, plus toothed whales
(Odontoceti). On the species tree this gives three origins: Rhinolophoidea, Yangochiroptera
and Odontoceti.

Pipeline (as validated on simulations):
  layers  log nucleotide rate, log protein rate, selection proxy (difference); pseudo-count of
          one substitution per alignment; mean-centred per branch; root edges excluded
  test    per layer GLS with a genome-learned covariance (pairwise-complete, missing branches
          allowed), leave-one-origin-out intersection-union test, ACAT over layers, BH FDR
  signatures  mash-style model with learned directions on pooled per-layer z-scores
  calibration the same test on control trait sets (non-echolocating clades matched to the
          three origins in size and age); every call for a control set is false

Usage: python -I scripts/echolocation_toga.py --bl /path/to/bl --out results/echolocation_toga
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
from mlconv.gls import gls_z_missing, learn_covariance_missing, leave_one_origin_out_p_missing  # noqa: E402
from mlconv.signatures import fit_signatures, pattern_active  # noqa: E402
from mlconv.stats import acat, bh  # noqa: E402
from mlconv.tree import (foreground_branches, from_newick, matched_clade_permute, origin_index,  # noqa: E402
                         origin_nodes)

LAYERS = ("rate_nucleotide", "rate_protein", "selection_proxy")
HEARING = ["SLC26A5", "TMC1", "PJVK", "CDH23", "PCDH15", "OTOF", "KCNQ4", "TECTA", "STRC", "CABP2",
           "OTOA", "LOXHD1", "ESPN", "GJB2", "MYO15A", "USH1C", "TMPRSS3", "SLC17A8"]


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bl", required=True, help="directory with master.nwk and branch_lengths.jsonl")
    ap.add_argument("--species", default=str(ROOT / "results" / "toga" / "species.tsv"))
    ap.add_argument("--coverage", required=True, help="TOGA coverage table with lineages")
    ap.add_argument("--out", default=str(ROOT / "results" / "echolocation_toga"))
    ap.add_argument("--controls", type=int, default=20)
    ap.add_argument("--q", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    out = Path(args.out)
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
    trait = np.array([("Chiroptera" in l and "Pteropodidae" not in l) or "Odontoceti" in l
                      for l in lineage])
    fg = foreground_branches(tree, trait)
    origin = origin_index(tree, trait)[keep]
    origins = origin_nodes(tree, trait)
    print(f"echolocators: {trait.sum()} species, {len(origins)} origins, {int(fg[keep].sum())} "
          f"foreground branches", flush=True)
    (out / "trait_species.txt").write_text("\n".join(n for n, t in zip(names, trait) if t) + "\n")

    t0 = time.time()
    Sig = [learn_covariance_missing(X[:, l]) for l in range(3)]
    p_final = final_test(X, Sig, origin)
    z = np.column_stack([gls_z_missing(X[:, l], Sig[l], fg[keep].astype(float)) for l in range(3)])
    rej = bh(p_final, args.q)
    print(f"echolocation: {rej.sum()} genes at q = {args.q} ({time.time() - t0:.0f}s)", flush=True)

    # signatures: genome-wide cross-layer null correlation from all genes' z-scores
    zc = np.nan_to_num(z)
    R = np.corrcoef(zc.T)
    sig = fit_signatures(zc, np.broadcast_to(R, (len(zc), 3, 3)).copy(), list(LAYERS))
    active = pattern_active(sig)
    res = pd.DataFrame({"gene": genes, "symbol": symbols, "p_final": p_final,
                        "q_final": st.false_discovery_control(np.clip(p_final, 1e-300, 1)),
                        **{f"z_{n}": z[:, i] for i, n in enumerate(LAYERS)},
                        "signature": ["+".join(n for n, a in zip(LAYERS, row) if a) for row in active]})
    res.loc[~rej, "signature"] = ""
    res.sort_values("p_final").to_csv(out / "echolocation_genes.csv", index=False)
    pd.Series({"_".join(n.split(":")[0:1]) + str(i): float(w) for i, ((n, _), w) in
               enumerate(zip(sig.patterns, sig.pattern_weights))}).to_csv(out / "signature_patterns.csv")
    print("\nHearing genes:")
    print(res[res.symbol.isin(HEARING)].sort_values("p_final").to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print("\nTop 20:")
    print(res.sort_values("p_final").head(20).to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    # calibration on matched control trait sets
    ctrl = matched_clade_permute(tree, origins, 10 * args.controls, rng, exclude=trait, size_tol=1)
    ctrl = np.unique(ctrl, axis=0)
    D = tree.tip_sets()
    rows = []
    for c in ctrl:
        tips = np.array([D[np.nonzero(c)[0], t].any() for t in range(tree.n_tips)])
        if len(origin_nodes(tree, tips)) != len(origins):
            continue
        oc = origin_index(tree, tips)[keep]
        pc = final_test(X, Sig, oc)
        rows.append(dict(control=len(rows), calls=int(bh(pc, args.q).sum()), min_p=float(pc.min()),
                         species=";".join(n for n, t in zip(names, tips) if t)))
        print(f"control {len(rows)}: {rows[-1]['calls']} calls", flush=True)
        pd.DataFrame(rows).to_csv(out / "control_calls.csv", index=False)
        if len(rows) == args.controls:
            break
    cc = pd.DataFrame(rows)
    summary = dict(genes=len(genes), species=tree.n_tips, echolocating_species=int(trait.sum()),
                   origins=len(origins), echolocation_calls=int(rej.sum()), control_sets=len(cc),
                   control_mean_calls=float(cc.calls.mean()) if len(cc) else None,
                   control_sets_with_any_call=float((cc.calls > 0).mean()) if len(cc) else None)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
