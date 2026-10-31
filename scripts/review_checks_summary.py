"""Summary of the analyses requested by the external review (run after scripts/run_review_checks.sh).

1. Held-out control traits: the first 20 control traits of each analysis (same seed) were used while the
   loss null was developed; traits 21-100 were not. Call rates are reported separately.
2. Composite-null tier: Benjamini-Hochberg over all loss-tested genes on the intersection-union P value
   (100,000 replicates per gene and origin) for the global-null calls, global-null P for the rest.
3. Call-set stability under covariance shrinkage, loss-model alternatives and the third coding.
4. Enrichment with and without collapsing gene families.
Writes results/review/summary.md and CSV tables next to it.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
RES, REV = ROOT / "results", ROOT / "results" / "review"
TRAITS = ["marine", "hypoxia_subterranean", "hypoxia_altitude", "hypoxia", "echolocation"]


def ci(k, n):
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return f"{100 * k / n:.0f}% ({100 * lo:.0f}-{100 * hi:.0f}%)"


def md(df):
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```\n" + df.to_string(index=False) + "\n```"


def calls(path):
    d = pd.read_csv(path)
    return d, set(d[d.call_loss == True].symbol)  # noqa: E712


out = ["# Review checks: summary", ""]
# 1. held-out controls
rows = []
for coding, pre in (("L+UL", "ctrl100_"), ("L only", "Lonly100_")):
    for t in TRAITS:
        f = RES / "audit" / f"{pre}{t}" / "control_calls.csv"
        if not f.exists():
            continue
        c = pd.read_csv(f)
        for part, sub in (("development (1-20)", c[c.control < 20]), ("held out (21-100)", c[c.control >= 20])):
            k, n = int((sub.calls_loss > 0).sum()), len(sub)
            rows.append(dict(coding=coding, trait=t, controls=part, n=n, any_loss_call=ci(k, n),
                             mean_false_loss_calls=round(sub.calls_loss.mean(), 3)))
h = pd.DataFrame(rows)
h.to_csv(REV / "heldout_controls.csv", index=False)
out += ["## 1. Development versus held-out control traits", "", md(h), ""]

# 2. composite-null tier
rows = []
for t in ("marine", "hypoxia_subterranean"):
    f = REV / f"iut1e5_{t}" / f"{t}_genes.csv"
    if not f.exists():
        continue
    a = pd.read_csv(f)
    a = a[a.p_loss.notna()]
    m = len(a)
    q = np.where(a.p_loss_iut.notna(), np.maximum(a.p_loss_iut.fillna(0), a.p_loss), a.p_loss)
    s = np.sort(q)
    ok = np.nonzero(s <= 0.1 * np.arange(1, m + 1) / m)[0]
    k = ok.max() + 1 if len(ok) else 0
    called = sorted(a.symbol[q <= 0.1 * k / m]) if k else []
    rows.append(dict(trait=t, loss_tested=m, global_calls=int(a.call_loss.sum()), composite_null_calls=k,
                     genes=", ".join(called), min_iut_p=float(np.nanmin(a.p_loss_iut))))
c2 = pd.DataFrame(rows)
c2.to_csv(REV / "composite_null_tier.csv", index=False)
out += ["## 2. Composite-null tier (100,000 replicates per gene and origin)", "", md(c2), ""]

# 3. call-set stability
rows = []
for d in sorted(p for p in REV.iterdir() if p.is_dir() and p.name not in ("flags",)):
    if d.name.startswith("iut1e5"):
        continue
    t = d.name.split("_", 1)[1]
    f = d / f"{t}_genes.csv"
    base = RES / f"{t}_toga" / f"{t}_genes.csv"
    if not (f.exists() and base.exists()):
        continue
    _, s = calls(f)
    _, b = calls(base)
    j = len(s & b) / len(s | b) if s | b else 1.0
    summ = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {}
    rows.append(dict(run=d.name, calls=len(s), main_calls=len(b), shared=len(s & b), jaccard=round(j, 2),
                     lost_from_main=", ".join(sorted(b - s)[:12]), new=", ".join(sorted(s - b)[:12]),
                     controls_any_loss_call=summ.get("control_sets_with_any_call_loss")))
st = pd.DataFrame(rows)
st.to_csv(REV / "callset_stability.csv", index=False)
out += ["## 3. Call-set stability", "", md(st) if len(st) else "(no runs found)", ""]

# 4. enrichment
rows = []
for src in ("tissue", "reactome", "go"):
    for tag in ("_review", "_review_collapsed"):
        f = RES / "extra" / f"enrichment_{src}{tag}.csv"
        if f.exists():
            e = pd.read_csv(f)
            for t, g in e.groupby("trait"):
                rows.append(dict(source=src, collapsed=tag.endswith("collapsed"), trait=t,
                                 significant_terms=int((g.q < 0.1).sum()),
                                 top=", ".join(g[g.q < 0.1].sort_values("q").name.head(5))))
en = pd.DataFrame(rows)
en.to_csv(REV / "enrichment_collapse.csv", index=False)
out += ["## 4. Enrichment with and without collapsing gene families", "",
        md(en) if len(en) else "(not run)", ""]
(REV / "summary.md").write_text("\n".join(out))
print("\n".join(out))
