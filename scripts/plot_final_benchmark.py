"""Manuscript Figure 2: calibration, false discoveries, power and signature accuracy.

Usage: python -I scripts/plot_final_benchmark.py results/final_benchmark.csv results/fig_final_benchmark.png
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
METHODS = [("naive", "Naive (RERconverge-style)", "#2a78d6"),
           ("permulation", "Control-lineage permutation", "#eb6834"),
           ("naive_fisher", "Naive, Fisher over layers", "#1baf7a"),
           ("brownian_lmm", "Brownian phylo-LMM", "#eda100"),
           ("gls", "Learned-covariance GLS", "#e87ba4"),
           ("final", "GLS + leave-one-origin-out (final)", "#008300")]
EFFECTS = ["weak", "moderate", "strong"]


def style(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis=grid_axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


df = pd.read_csv(sys.argv[1])
fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), constrained_layout=True)
conf = sorted(df.confounding.unique())
x = np.arange(len(conf))
w = 0.13

# a: false calls per genome with no convergence (one per tree/confounding; weak rows suffice)
ax = axes[0, 0]
null = df[df.effect == "weak"].groupby(["confounding", "method"]).false_calls_null.mean()
for i, (m, lab, c) in enumerate(METHODS):
    v = np.array([null.get((cf, m), np.nan) for cf in conf])
    ax.scatter(x + (i - 2.5) * w, v + 1, color=c, s=26, zorder=3, label=lab)
ax.set_yscale("log")
ax.set_xticks(x, [f"{c:g}" for c in conf])
ax.set_xlabel("Phylogenetic confounding (rate-jump intensity)", fontsize=9, color=INK)
ax.set_ylabel("False calls + 1 (genome with no convergence)", fontsize=9, color=INK)
ax.set_title("a  False calls when nothing converged", loc="left", fontsize=10, color=INK)
style(ax)

# b: realised FDR when 2% of genes converged, by confounding
ax = axes[0, 1]
fdr = df.groupby(["confounding", "method"]).fdr.mean()
for i, (m, lab, c) in enumerate(METHODS):
    v = np.array([fdr.get((cf, m), np.nan) for cf in conf])
    ax.scatter(x + (i - 2.5) * w, v, color=c, s=26, zorder=3)
ax.axhline(0.10, color=INK, lw=1, ls="--")
ax.text(2.45, 0.12, "target 0.10", fontsize=8, color=MUTED, ha="right")
ax.set_xticks(x, [f"{c:g}" for c in conf])
ax.set_ylim(-0.02, 1.0)
ax.set_xlabel("Phylogenetic confounding (rate-jump intensity)", fontsize=9, color=INK)
ax.set_ylabel("Realised FDR at q = 0.10", fontsize=9, color=INK)
ax.set_title("b  False discoveries (2% of genes convergent)", loc="left", fontsize=10, color=INK)
style(ax)

# c: power of the final test by kind of convergence and effect size
ax = axes[1, 0]
fin = df[df.method == "final"].groupby("effect")
kinds = [("power_both", "Selection + rate", "-", "o"),
         ("power_omega", "Selection shift (ω)", "--", "s"),
         ("power_rate", "Rate shift only", ":", "^")]
for col, lab, ls, mk in kinds:                     # final method only: its colour, styles for kinds
    v = fin[col].mean().reindex(EFFECTS)
    ax.plot(range(3), v.values, color="#008300", lw=2, ls=ls, marker=mk, ms=6, label=lab)
ax.set_xticks(range(3), ["weak\nω×1.5, rate×1.2", "moderate\nω×2, rate×1.35", "strong\nω×3, rate×1.5"])
ax.set_ylim(0, 1.04)
ax.set_ylabel("Power at q = 0.10 (final test)", fontsize=9, color=INK)
ax.set_title("c  Final test: convergent genes found, by kind", loc="left", fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=8, loc="lower right")
style(ax)

# d: signature accuracy of the final pipeline
ax = axes[1, 1]
sig = df[df.method == "final"].groupby("effect").signature_accuracy
m = sig.mean().reindex(EFFECTS)
lo = sig.min().reindex(EFFECTS)
hi = sig.max().reindex(EFFECTS)
ax.errorbar(range(3), m.values, yerr=[m.values - lo.values, hi.values - m.values], color="#008300",
            lw=2, marker="o", ms=5, capsize=4)
ax.set_xticks(range(3), ["weak", "moderate", "strong"])
ax.set_ylim(0, 1.04)
ax.set_ylabel("Exact signature accuracy", fontsize=9, color=INK)
ax.set_xlabel("Effect size", fontsize=9, color=INK)
ax.set_title("d  Final pipeline: correct convergence signature", loc="left", fontsize=10, color=INK)
ax.text(0.02, 0.06, "mean over 9 tree × confounding settings; bars: min-max", transform=ax.transAxes,
        fontsize=7.5, color=MUTED)
style(ax)

handles = [plt.Line2D([], [], color=c, marker="o", lw=0, ms=6) for _, _, c in METHODS]
fig.legend(handles, [lab for _, lab, _ in METHODS], loc="outside lower center", ncol=3, frameon=False,
           fontsize=8.5)
fig.savefig(sys.argv[2], dpi=200, facecolor="white")
print("wrote", sys.argv[2])
