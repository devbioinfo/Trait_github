"""Sequence-level validation: simulate codon alignments with IQ-TREE AliSim, re-estimate
branch lengths with IQ-TREE, build layer statistics, and test every method.

Per gene g (codon model GY, omega_g ~ lognormal around 0.15, length 300-1500 codons):
  * phylogenetic confounding: K_g ~ Poisson(confounding * 3) random clades get a
    clade-wide rate multiplier ~ lognormal(0, 0.7);
  * a gene-wide rate multiplier ~ lognormal(0, 0.5);
  * convergent genes (fraction --frac) receive, on foreground branches only,
      "omega": omega x omega_mult  (selection shift; branch lengths unchanged)
      "rate":  branch length x rate_mult  (acceleration of all substitutions)
      "both":  both of the above.
Estimation (topology fixed to the true tree): GTR on nucleotides and LG on the translated
protein (the simulation has no among-site rate variation, so +G is not fitted). Layers: log nucleotide length, log protein length, and their
difference (a dN/dS-like proxy), each made relative to the genome-wide branch median.
The two root branches are merged in an unrooted estimate and are dropped.

Usage: python -I scripts/alisim_validation.py --out results/alisim.csv
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlconv.gls import gls_z, leave_one_origin_out_p, learn_covariance  # noqa: E402
from mlconv.lmm import PhyloLMM  # noqa: E402
from mlconv.stats import acat  # noqa: E402
from scipy import stats as st  # noqa: E402
from mlconv.stats import bh, run_all_methods, run_lmm_methods  # noqa: E402
from mlconv.tree import (branch_splits, clade_permute, foreground_branches,  # noqa: E402
                         origin_index, origin_sizes, parse_newick_splits, pick_convergent_clades,
                         to_newick, yule_tree)

LAYERS = ("rate_nucleotide", "rate_protein", "selection_proxy")
IQTREE = shutil.which("iqtree2") or "iqtree2"
CODON_TABLE = {
    a + b + c: aa for (a, b, c), aa in zip(
        ((x, y, z) for x in "TCAG" for y in "TCAG" for z in "TCAG"),
        "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG")
}


def translate(seq: str) -> str:
    return "".join(CODON_TABLE.get(seq[i:i + 3], "X") for i in range(0, len(seq) - 2, 3))


def read_phylip(path: Path) -> dict[str, str]:
    lines = path.read_text().split("\n")[1:]
    return {ln.split()[0]: ln.split()[1] for ln in lines if ln.strip()}


def write_phylip(path: Path, seqs: dict[str, str]) -> None:
    n, L = len(seqs), len(next(iter(seqs.values())))
    path.write_text(f"{n} {L}\n" + "".join(f"{k} {v}\n" for k, v in seqs.items()))


def run_iqtree(args: list[str], cwd: Path) -> None:
    subprocess.run([IQTREE, *args, "-nt", "1", "--redo", "-quiet"], cwd=cwd,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def simulate_and_estimate(job: dict) -> tuple[np.ndarray, np.ndarray]:
    """Returns estimated nucleotide and protein branch lengths in canonical-split order."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "sim.nwk").write_text(job["sim_newick"] + "\n")
        (tmp / "true.nwk").write_text(job["true_newick"] + "\n")
        run_iqtree(["--alisim", "aln", "-t", "sim.nwk", "-m", job["model"],
                    "--length", str(3 * job["codons"]), "-seed", str(job["seed"])], tmp)
        nt = read_phylip(tmp / "aln.phy")
        write_phylip(tmp / "prot.phy", {k: translate(v) for k, v in nt.items()})
        run_iqtree(["-s", "aln.phy", "-te", "true.nwk", "-m", "GTR", "--prefix", "nt"], tmp)
        run_iqtree(["-s", "prot.phy", "-te", "true.nwk", "-m", "LG", "--prefix", "aa"], tmp)
        n_tips = job["n_tips"]
        bl_nt = parse_newick_splits((tmp / "nt.treefile").read_text().strip(), n_tips)
        bl_aa = parse_newick_splits((tmp / "aa.treefile").read_text().strip(), n_tips)
    keys = job["keys"]
    return (np.array([bl_nt.get(k, 0.0) for k in keys]),
            np.array([bl_aa.get(k, 0.0) for k in keys]))


def make_jobs(tree, fg, n_genes, rng, *, frac, confounding, omega_mult, rate_mult, height, kappa=2.0):
    D = tree.tip_sets()
    internal = np.arange(tree.n_tips, tree.root)
    keys = branch_splits(tree)
    jobs, truth, kinds = [], [], []
    for g in range(n_genes):
        lengths = tree.length.copy() * rng.lognormal(0.0, 0.5)
        for v in rng.choice(internal, size=rng.poisson(confounding * 3), replace=False):
            below = np.nonzero(~(D[: tree.n_branches] & ~D[v]).any(1))[0]   # branches inside clade v
            lengths[below] *= rng.lognormal(0.0, 0.7)
        omega = float(np.clip(rng.lognormal(np.log(0.15), 0.6), 0.01, 0.9))
        conv = rng.random() < frac
        kind = rng.choice(["omega", "rate", "both"], p=[0.4, 0.3, 0.3]) if conv else "none"
        ann = {}
        if kind in ("rate", "both"):
            lengths[np.nonzero(fg)[0]] *= rate_mult
        if kind in ("omega", "both"):
            w_fg = omega * omega_mult
            ann = {int(b): f"[&model=GY{{{w_fg:.4f},{kappa}}}]" for b in np.nonzero(fg)[0]}
        jobs.append(dict(sim_newick=to_newick(tree, height, lengths, ann),
                         true_newick=to_newick(tree, height),
                         model=f"GY{{{omega:.4f},{kappa}}}",
                         codons=int(np.clip(rng.lognormal(np.log(500), 0.5), 300, 1500)),
                         seed=int(rng.integers(1, 2**31 - 1)), n_tips=tree.n_tips, keys=keys))
        truth.append(conv)
        kinds.append(kind)
    return jobs, np.array(truth), np.array(kinds)


def layers_from_lengths(nt: np.ndarray, aa: np.ndarray, codons: np.ndarray) -> np.ndarray:
    """Log branch lengths with a pseudo-count of one substitution per alignment
    (1/sites), which stabilises the variance of short branches; then mean-centred
    per branch across genes (relative rates)."""
    ln_nt = np.log(nt + 1.0 / (3 * codons)[:, None])
    ln_aa = np.log(aa + 1.0 / codons[:, None])
    X = np.stack([ln_nt, ln_aa, ln_aa - ln_nt], axis=1)              # (genes, layers, branches)
    return X - X.mean(0, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/alisim.csv")
    ap.add_argument("--tips", type=int, default=80)
    ap.add_argument("--genes", type=int, default=1000)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--frac", type=float, default=0.02)
    ap.add_argument("--origins", type=int, default=3)
    ap.add_argument("--min-clade", type=int, default=3)
    ap.add_argument("--max-clade", type=int, default=6)
    ap.add_argument("--confounding", default="0,1")
    ap.add_argument("--omega-mult", type=float, default=3.0)
    ap.add_argument("--rate-mult", type=float, default=1.5)
    ap.add_argument("--height", type=float, default=0.8, help="root-to-tip codon substitutions")
    ap.add_argument("--null-draws", type=int, default=40)
    ap.add_argument("--perms", type=int, default=300)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--save-layers", default="")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    for rep in range(args.reps):
        tree = yule_tree(args.tips, rng)
        trait = pick_convergent_clades(tree, n_origins=args.origins,
                                       size_range=(args.min_clade, args.max_clade), rng=rng)
        fg = foreground_branches(tree, trait)
        origin = origin_index(tree, trait)
        keep = tree.parent[: tree.n_branches] != tree.root
        nulls = clade_permute(tree, origin_sizes(tree, trait), max(args.perms, args.null_draws),
                              rng, exclude=trait)
        lmm = PhyloLMM(tree, keep=keep)
        for conf in map(float, args.confounding.split(",")):
            for frac, label in ((0.0, "no_convergence"), (args.frac, "convergent")):
                t0 = time.time()
                jobs, truth, kinds = make_jobs(tree, fg, args.genes, rng, frac=frac, confounding=conf,
                                               omega_mult=args.omega_mult, rate_mult=args.rate_mult,
                                               height=args.height)
                with Pool(args.workers) as pool:
                    est = pool.map(simulate_and_estimate, jobs, chunksize=8)
                nt = np.array([e[0] for e in est])
                aa = np.array([e[1] for e in est])
                codons = np.array([j["codons"] for j in jobs])
                X = layers_from_lengths(nt, aa, codons)[:, :, keep]
                if args.save_layers:
                    np.savez_compressed(f"{args.save_layers}_rep{rep}_conf{conf}_{label}.npz",
                                        X=X, truth=truth, kinds=kinds, fg=fg[keep], nt=nt, aa=aa,
                                        codons=codons, keep=keep)
                fk, nk = fg[keep], nulls[:, keep]
                pv = run_all_methods(X, fk, nk[: args.perms])
                pv = {k: v for k, v in pv.items() if ":" not in k or k == "naive:0"}
                lm = run_lmm_methods(lmm, X, fk, nk[: args.null_draws])
                lm.pop("_layer_post")
                pv.update({k: v for k, v in lm.items() if k in ("lmm:0", "lmm:1", "lmm:2", "lmm_acat", "ebcm")})
                ok = origin[keep]
                p_pool, p_looo = [], []
                for l in range(X.shape[1]):
                    S = learn_covariance(X[:, l])
                    p_pool.append(st.norm.sf(gls_z(X[:, l], S, fk[None])[:, 0]))
                    p_looo.append(leave_one_origin_out_p(X[:, l], S, ok))
                for l in range(X.shape[1]):
                    pv[f"gls:{l}"] = p_pool[l]
                    pv[f"gls_looo:{l}"] = p_looo[l]
                pv["gls_acat"] = acat(np.column_stack(p_pool))
                pv["gls_looo_acat"] = acat(np.column_stack(p_looo))
                for name, p in pv.items():
                    rej = bh(p, 0.1)
                    row = dict(rep=rep, confounding=conf, scenario=label, method=name,
                               fpr=(p[~truth] < 0.05).mean(),
                               fdr=(rej & ~truth).sum() / max(1, rej.sum()),
                               power=(rej & truth).sum() / max(1, truth.sum()))
                    for kind in ("omega", "rate", "both"):
                        m = kinds == kind
                        row[f"power_{kind}"] = (rej & m).sum() / max(1, m.sum())
                    rows.append(row)
                print(f"rep={rep} conf={conf} {label}: {time.time() - t0:.0f}s", flush=True)
                pd.DataFrame(rows).to_csv(args.out, index=False)

    df = pd.DataFrame(rows)
    df["method"] = df["method"].replace({"naive:0": "naive:rate_nucleotide",
                                         **{f"{m}:{i}": f"{m}:{n}" for m in ("lmm", "gls", "gls_looo")
                                            for i, n in enumerate(LAYERS)}})
    df.to_csv(args.out, index=False)
    pd.set_option("display.width", 200)
    for m in ("fpr", "fdr", "power"):
        print(f"\n== {m}")
        print(df.groupby(["confounding", "scenario", "method"])[m].mean().round(3)
              .unstack("method").to_string())


if __name__ == "__main__":
    main()
