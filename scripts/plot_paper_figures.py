"""Figures for the paper from the five TOGA analyses and the loss-layer benchmark.

fig_traits.png         (a) calls per trait and layer; (b) calibration on 20 matched control
                       trait sets per analysis (share of sets with any call; nominal 10%)
fig_loss_benchmark.png power and false discovery proportion of the loss layer on simulated
                       genomes, by number of origins and penetrance, against two baselines

Usage: python -I scripts/plot_paper_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
# validated categorical slots 1-3 (light mode); text in neutral inks, never series colours
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TRAITS = [("marine", "Marine"), ("hypoxia_subterranean", "Hypoxia,\nsubterranean"),
          ("hypoxia", "Hypoxia,\ncombined"), ("hypoxia_altitude", "Hypoxia,\nhigh altitude"),
          ("echolocation", "Echolocation")]
LAYERS = [("rate", "Rate", BLUE), ("loss", "Gene loss", ORANGE), ("final", "Combined", AQUA)]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})


def summaries():
    rows = []
    for key, label in TRAITS:
        s = json.loads((RES / f"{key}_toga" / "summary.json").read_text())
        for lay, _, _ in LAYERS:
            rows.append(dict(trait=label, layer=lay, origins=s["origins"], calls=s[f"calls_{lay}"],
                             any_call=s[f"control_sets_with_any_call_{lay}"],
                             mean_calls=s[f"control_mean_calls_{lay}"]))
    return pd.DataFrame(rows)


def fig_traits():
    d = summaries()
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw=dict(wspace=0.75))
    y = np.arange(len(TRAITS))
    h = 0.26
    for i, (lay, name, col) in enumerate(LAYERS):
        v = d[d.layer == lay].calls.to_numpy()
        a.barh(y + (i - 1) * h, v, height=h - 0.04, color=col, label=name)
        for yy, vv in zip(y + (i - 1) * h, v):
            a.text(vv + 0.8, yy, str(vv), va="center", fontsize=7, color=INK2)
    a.set_yticks(y, [t for _, t in TRAITS])
    a.invert_yaxis()
    a.set_xlabel("Genes called (q = 0.1)")
    a.xaxis.grid(True, color=GRID, lw=0.6)
    a.set_axisbelow(True)
    a.legend(frameon=False, fontsize=8, loc="lower right")
    a.set_title("a  Convergent genes", loc="left", fontsize=10, color=INK)
    for i, (lay, name, col) in enumerate(LAYERS):
        v = 100 * d[d.layer == lay].any_call.to_numpy()
        b.scatter(v, y + (i - 1) * 0.22, s=36, color=col, edgecolor="white", linewidth=1.5, label=name, zorder=3)
    b.axvline(10, color=INK2, lw=1, ls="--")
    b.set_yticks(y, [t for _, t in TRAITS])
    b.set_ylim(len(TRAITS) - 0.5, -0.95)
    b.text(10.5, -0.72, "nominal 10%", fontsize=7, color=INK2, va="center")
    b.set_xlim(-1.5, 25)
    b.set_xlabel("Control trait sets with any call (%)")
    b.xaxis.grid(True, color=GRID, lw=0.6)
    b.set_axisbelow(True)
    b.set_title("b  Calibration, 20 control sets each", loc="left", fontsize=10, color=INK)
    fig.savefig(RES / "fig_traits.png", dpi=300, bbox_inches="tight", facecolor="white")
    print("written", RES / "fig_traits.png")


def fig_loss_benchmark():
    path = RES / "loss_benchmark.csv"
    if not path.exists():
        return
    d = pd.read_csv(path)
    methods = [("simulation", "This method", BLUE), ("gls", "GLS, uncorrected", ORANGE),
               ("fisher", "Fisher exact, no phylogeny", AQUA)]
    origins = sorted(d.origins.unique())
    fig, axes = plt.subplots(2, len(origins), figsize=(2.4 * len(origins) + 0.6, 4.6), sharey="row",
                             gridspec_kw=dict(hspace=0.45, wspace=0.12))
    axes = np.atleast_2d(axes)
    sig = d[d.penetrance > 0]
    for j, k in enumerate(origins):
        for m, name, col in methods:
            g = sig[(sig.origins == k) & (sig.method == m)].groupby("penetrance")
            for row, col_name in ((0, "power"), (1, "fdp")):
                mean, lo, hi = g[col_name].mean(), g[col_name].min(), g[col_name].max()
                ax = axes[row, j]
                ax.plot(mean.index, mean.values, color=col, lw=2, marker="o", ms=5,
                        markeredgecolor="white", markeredgewidth=1.2, label=name)
                ax.vlines(mean.index, lo.values, hi.values, color=col, lw=1)
        axes[0, j].set_title(f"{k} origins", fontsize=10, color=INK)
        axes[1, j].axhline(0.1, color=INK2, lw=1, ls="--")
        for row in (0, 1):
            axes[row, j].set_xticks([0.5, 0.75, 1.0])
            axes[row, j].yaxis.grid(True, color=GRID, lw=0.6)
            axes[row, j].set_axisbelow(True)
        axes[1, j].set_xlabel("Penetrance (loss per origin)")
    axes[0, 0].set_ylabel("Power")
    axes[1, 0].set_ylabel("False discovery proportion")
    axes[0, 0].set_ylim(0, 1)
    axes[1, 0].set_ylim(0, 0.6)
    axes[1, 0].text(1.0, 0.085, "q = 0.1", fontsize=7, color=INK2, ha="right", va="top")
    axes[0, -1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.savefig(RES / "fig_loss_benchmark.png", dpi=300, bbox_inches="tight", facecolor="white")
    print("written", RES / "fig_loss_benchmark.png")


if __name__ == "__main__":
    fig_traits()
    fig_loss_benchmark()
