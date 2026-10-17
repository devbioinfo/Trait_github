"""Figure: power and false discoveries vs number of independent origins.

Usage: python -I scripts/plot_origins_power.py results/origins_power.csv results/fig_origins_power.png
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xticks([2, 3, 4, 5, 6])
    ax.set_xlabel("Independent origins of the trait", fontsize=9, color=INK)


df = pd.read_csv(sys.argv[1])
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)

# a: power of the final test by kind of convergence
ax = axes[0]
final = df[df.method == "gls_looo_acat"].groupby("origins")
series = [("power_omega", "Selection shift (ω)", "#2a78d6"),
          ("power_both", "Selection + rate", "#eb6834"),
          ("power_rate", "Rate shift only", "#1baf7a")]
for col, lab, c in series:
    m = final[col].mean()
    ax.plot(m.index, m.values, color=c, lw=2, marker="o", ms=5, label=lab)
ax.set_ylim(0, 1.04)
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.set_ylabel("Power at q = 0.10", fontsize=9, color=INK)
ax.set_title("a  Final test: convergent genes found", loc="left", fontsize=10, color=INK)
style(ax)

# b: false discovery rate by method
ax = axes[1]
meths = [("naive:rate_nucleotide", "Naive (RERconverge-style)", "#6250d6"),
         ("gls_acat", "Learned covariance", "#eda100"),
         ("gls_looo_acat", "+ leave-one-origin-out (final)", "#008300")]
for m, lab, c in meths:
    s = df[df.method == m].groupby("origins").fdr.mean()
    ax.plot(s.index, s.values, color=c, lw=2, marker="o", ms=5, label=lab)
ax.axhline(0.10, color=INK, lw=1, ls="--")
ax.text(2, 0.115, "target 0.10", fontsize=8, color=MUTED)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Realised FDR at q = 0.10", fontsize=9, color=INK)
ax.set_title("b  False discoveries", loc="left", fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=8, loc="center right", bbox_to_anchor=(1.0, 0.55))
style(ax)

fig.text(0.01, -0.03, "AliSim codon simulations on Yule trees of 200 species; 3,000 genes, 2% convergent; "
         "3 trees, nested origins.", fontsize=7.5, color=MUTED)
fig.savefig(sys.argv[2], dpi=200, facecolor="white", bbox_inches="tight")
print("wrote", sys.argv[2])
