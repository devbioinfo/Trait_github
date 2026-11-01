"""Summary of the second-audit verification runs (results/audit2): conservative composite-null BH,
Monte Carlo p-values with uncertainty, an evidence table for the surviving genes, coding comparison,
the full BH sequence for marine mammals, and the simulation check of the single-origin test.

Conservative composite-null BH: every loss-tested gene gets p' = its single-origin (intersection-union)
Monte Carlo p-value if it was called under the global null, and p' = 1 otherwise. Because p' is never
smaller than the gene's single-origin p-value would be, p' is a valid p-value whatever the screening
step did, and BH applied to p' over all loss-tested genes needs no further argument about the screen.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "results" / "audit2"
Q = 0.1
FOCUS = ["CRYGS", "LGSN", "MPP4", "CABP5", "LCTL", "BFSP1", "ROM1", "RGS11", "LACTBL1", "RSKR", "TPD52L3"]


def md(df, index=False):
    """Markdown table without the optional tabulate package."""
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    fmt = lambda v: (f"{v:.3g}" if isinstance(v, float) else str(v))
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join(lines)


def bh_adjust(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    adj = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(adj, 1)
    rank = np.empty(n, int)
    rank[o] = np.arange(1, n + 1)
    return out, rank


def cp_upper(k, n, level=0.95):
    return beta.ppf(level, k + 1, n - k) if k < n else 1.0


def analyse(run, label):
    f = next((A / run).glob("*_genes.csv"), None)
    if f is None:
        return None, None
    g = pd.read_csv(f)
    g = g[g.p_loss.notna()].copy()
    m = len(g)
    kcols = sorted(c for c in g.columns if c.startswith("k_loss_iut_origin"))
    called = g.call_loss.astype(bool)
    g["p_prime"] = np.where(called & g.p_loss_iut_mc.notna(), g.p_loss_iut_mc, 1.0)
    g["p_adj"], g["rank"] = bh_adjust(g.p_prime)
    g["discovery"] = g.p_adj <= Q
    # Monte Carlo uncertainty: one-sided 95% upper bound of the largest per-origin p-value
    def upper(r):
        if not kcols or pd.isna(r.iut_sims):
            return np.nan
        n = int(r.iut_sims)
        return max(cp_upper(int(r[c]), n) for c in kcols if pd.notna(r[c]))
    chk = g.p_loss_iut_mc.notna()
    g["p_mc_upper95"] = np.nan
    g.loc[chk, "p_mc_upper95"] = g[chk].apply(upper, axis=1)
    pu = np.where(called & chk, g.p_mc_upper95.fillna(1.0), 1.0)
    g["discovery_at_upper95"] = bh_adjust(pu)[0] <= Q
    s = dict(analysis=label, loss_tested=m, global_calls=int(called.sum()),
             iut_replicates=int(g.iut_sims.dropna().iloc[0]) if g.iut_sims.notna().any() else None,
             discoveries=int(g.discovery.sum()),
             genes=", ".join(g[g.discovery].sort_values("p_prime").symbol),
             discoveries_at_mc_upper95=int(g.discovery_at_upper95.sum()),
             first_bh_threshold=Q / m)
    return g, s


rows, tabs = [], {}
runs = [("iut_primary_subterranean", "Subterranean, L+UL (primary)"),
        ("iut_Lonly_subterranean", "Subterranean, L only (UL missing)"),
        ("iut_ULintact_subterranean", "Subterranean, UL as intact"),
        ("iut_primary_marine", "Marine, L+UL (primary)")]
for run, label in runs:
    g, s = analyse(run, label)
    if g is not None:
        tabs[run] = g
        rows.append(s)
summary = pd.DataFrame(rows)
summary.to_csv(A / "composite_null_bh.csv", index=False)

# evidence table: the focus genes and every discovery, in every coding
ev = pd.read_csv(ROOT / "results/extra/loss_evidence_by_gene.csv")
ev = ev[ev.trait == "Subterranean"].set_index("gene")
genes = set(FOCUS)
for run, g in tabs.items():
    if "subterranean" in run:
        genes |= set(g[g.discovery].symbol)
erows = []
for gene in sorted(genes):
    r = {"gene": gene}
    for run, label in runs[:3]:
        g = tabs.get(run)
        if g is None:
            continue
        x = g[g.symbol == gene]
        tag = run.replace("iut_", "").replace("_subterranean", "")
        if x.empty:
            r[f"{tag}_status"] = "not loss-tested"
            continue
        x = x.iloc[0]
        kc = sorted(c for c in g.columns if c.startswith("k_loss_iut_origin"))
        r[f"{tag}_global_p"] = x.p_loss
        r[f"{tag}_global_call"] = bool(x.call_loss)
        r[f"{tag}_iut_p_mc"] = x.p_loss_iut_mc
        r[f"{tag}_iut_p_upper95"] = x.p_mc_upper95
        r[f"{tag}_exceedances_by_origin"] = ";".join(str(int(x[c])) for c in kc if pd.notna(x[c]))
        r[f"{tag}_replicates"] = x.iut_sims
        r[f"{tag}_bh_rank"] = int(x["rank"]) if bool(x.call_loss) else None
        r[f"{tag}_bh_adjusted"] = x.p_adj
        r[f"{tag}_discovery"] = bool(x.discovery)
        r[f"{tag}_trait_species_lost"] = x.trait_species_lost
    if gene in ev.index:
        e = ev.loc[gene]
        r.update(origins_with_loss=e.origins_with_loss, origins_lesion_supported=e.origins_lesion_supported,
                 species_L=e.species_L, species_UL=e.species_UL, lesions_stop=e.lesions_stop,
                 lesions_frameshift=e.lesions_frameshift, lesions_splice=e.lesions_splice,
                 lesions_deleted_exon=e.lesions_deleted_exon)
    erows.append(r)
evid = pd.DataFrame(erows)
evid.to_csv(A / "composite_null_evidence_table.csv", index=False)

# marine: the full BH sequence of the conservative p-values (why there are no discoveries)
mar = tabs.get("iut_primary_marine")
if mar is not None:
    s = mar.sort_values("p_prime").head(15)[["symbol", "p_prime", "rank", "p_adj"]].copy()
    s["bh_threshold"] = s["rank"] * Q / len(mar)
    s["passes_at_its_rank"] = s.p_prime <= s.bh_threshold
    s.to_csv(A / "marine_bh_sequence.csv", index=False)

# simulation check
sims = sorted(A.glob("composite_null_sim_*_summary.csv"))

with open(A / "summary.md", "w") as f:
    f.write("# Second-audit verification\n\n## Conservative composite-null BH (untested genes p = 1)\n\n")
    f.write(md(summary) if len(summary) else "no runs found")
    f.write("\n\n## Evidence table\n\n" + (md(evid.set_index("gene").T, index=True) if len(evid) else "none"))
    if mar is not None:
        f.write("\n\n## Marine BH sequence (15 smallest)\n\n" + md(pd.read_csv(A / "marine_bh_sequence.csv")))
    for s in sims:
        f.write(f"\n\n## Simulation check: {s.stem}\n\n" + md(pd.read_csv(s)))
    f.write("\n")
print((A / "summary.md").read_text())
