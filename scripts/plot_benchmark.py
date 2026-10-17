"""Figure: calibration and power of convergence tests (benchmark v2).

Usage: python -I scripts/plot_benchmark.py results/benchmark_v2.csv results/fig_benchmark.png
"""
from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Reference categorical palette (light mode), fixed order.
METHODS = [
    ("naive:relative_rate", "Naive, single layer", "#2a78d6"),
    ("naive_fisher", "Naive, Fisher", "#eb6834"),
    ("perm_fisher", "Permulation, Fisher", "#1baf7a"),
    ("lmm_fisher", "Phylo-LMM, Fisher", "#eda100"),
    ("lmm_acat", "Phylo-LMM, ACAT", "#e87ba4"),
    ("ebcm", "Phylo-LMM + EBCM (this work)", "#008300"),
]
SCEN_ORDER = ["baseline", "heavy_tails", "lineage", "tree_error", "sparse", "dense"]
SCEN_LABEL = {"baseline": "Baseline", "heavy_tails": "Heavy tails", "lineage": "Lineage shift",
              "tree_error": "Tree error", "sparse": "Sparse signal", "dense": "Dense signal"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def main(path: str, out: str) -> None:
    df = pd.read_csv(path)
    df["scen"] = df["scenario"]
    df.loc[df.scenario == "baseline", "scen"] = None
    base = df[df.scenario == "baseline"].copy()
    # confounding level is not stored for baseline rows; recover from row order per rep
    other = df[df.scenario != "baseline"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    x = range(len(SCEN_ORDER))
    width = 0.13

    # a: false positive rate under the global null (effect 0)
    ax = axes[0]
    null = pd.concat([base.assign(scenario="baseline"), other])
    null = null[null.effect == 0].groupby(["scenario", "method"]).fpr.mean()
    for i, (m, lab, col) in enumerate(METHODS):
        if m == "ebcm":
            continue
        vals = [null.get((s, m), float("nan")) for s in SCEN_ORDER]
        ax.scatter([xi + (i - 2.5) * width for xi in x], vals, s=22, color=col, label=lab, zorder=3)
    ax.axhline(0.05, color=INK, lw=1, ls="--")
    ax.set_xticks(list(x), [SCEN_LABEL[s] for s in SCEN_ORDER], rotation=30, ha="right")
    ax.set_ylabel("False positive rate at α = 0.05", fontsize=9, color=INK)
    ax.set_title("a  Calibration under no convergence", loc="left", fontsize=10, color=INK)
    style(ax)

    # b: realised FDR at q = 0.1 when convergence is present
    ax = axes[1]
    sig = pd.concat([base, other])
    sig = sig[sig.effect > 0].groupby(["scenario", "method"]).fdr.mean()
    for i, (m, lab, col) in enumerate(METHODS):
        vals = [sig.get((s, m), float("nan")) for s in SCEN_ORDER]
        ax.scatter([xi + (i - 2.5) * width for xi in x], vals, s=22, color=col, zorder=3)
    ax.axhline(0.10, color=INK, lw=1, ls="--")
    ax.set_xticks(list(x), [SCEN_LABEL[s] for s in SCEN_ORDER], rotation=30, ha="right")
    ax.set_ylabel("Realised FDR at q = 0.10", fontsize=9, color=INK)
    ax.set_title("b  False discovery rate", loc="left", fontsize=10, color=INK)
    style(ax)

    # c: power vs effect size, baseline scenario
    ax = axes[2]
    pw = base.groupby(["effect", "method"]).power.mean().unstack()
    for m, lab, col in METHODS:
        if m in pw:
            ax.plot(pw.index, pw[m], color=col, lw=2, marker="o", ms=4)
    ax.set_xlabel("Effect size (SD units)", fontsize=9, color=INK)
    ax.set_ylabel("Power at q = 0.10", fontsize=9, color=INK)
    ax.set_title("c  Power (baseline scenarios)", loc="left", fontsize=10, color=INK)
    ax.set_ylim(0, 1.02)
    style(ax)

    handles = [plt.Line2D([], [], color=c, marker="o", lw=2, ms=5) for _, _, c in METHODS]
    fig.legend(handles, [lab for _, lab, _ in METHODS], loc="outside lower center", ncol=6,
               frameon=False, fontsize=8)
    fig.savefig(out, dpi=200, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
