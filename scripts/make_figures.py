"""Numbered manuscript figures (PNG at 300 dpi and PDF) in results/figures/.

Figure 1  method overview (schematic)
Figure 2  rate layers on simulated codon alignments (calibration, power, signatures, origins)
Figure 3  gene-loss layer: simulation benchmark and calibration of loss nulls on real genomes
Figure 4  genome-wide results for three traits: calls, calibration, and the convergent losses
Supplementary Figure 1  comparison with RERconverge

Usage: python -I scripts/make_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mlconv.tree import foreground_branches, from_newick  # noqa: E402

RES = ROOT / "results"
OUT = RES / "figures"
OUT.mkdir(exist_ok=True)

# validated categorical palette (light), in fixed order; text in neutral inks
C1, C2, C3, C4, C5, C6 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
INK, INK2, MUTED, GRID, PALE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#f1f0ec"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 9,
                     "legend.fontsize": 7, "legend.frameon": False})


def panel(ax, letter, title=""):
    ax.set_title(f"{letter}  {title}" if title else letter, loc="left", fontsize=9, color=INK,
                 fontweight="bold" if not title else "normal")


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("written", OUT / f"{name}.png")


# ---------------------------------------------------------------------------------------
# tree drawing
# ---------------------------------------------------------------------------------------

def tree_xy(tree):
    """x = distance from root, y = tip order (tips 0..n-1), internal y = mean of children."""
    x = np.zeros(tree.n_nodes)
    for v in range(tree.n_nodes - 2, -1, -1):
        x[v] = x[tree.parent[v]] + tree.length[v]
    y = np.zeros(tree.n_nodes)
    y[: tree.n_tips] = np.arange(tree.n_tips)
    for v in range(tree.n_tips, tree.n_nodes):
        y[v] = y[tree.parent == v].mean()
    return x, y


def draw_tree(ax, tree, colors=None, lw=1.6, grey=MUTED):
    """colors: dict branch -> colour (others grey). Rectangular layout."""
    x, y = tree_xy(tree)
    colors = colors or {}
    for v in range(tree.n_nodes - 1):
        p = tree.parent[v]
        c = colors.get(v, grey)
        ax.plot([x[p], x[v]], [y[v], y[v]], color=c, lw=lw, solid_capstyle="round", zorder=2)
    for v in range(tree.n_tips, tree.n_nodes):
        ch = np.nonzero(tree.parent == v)[0]
        ax.plot([x[v], x[v]], [y[ch].min(), y[ch].max()], color=grey, lw=lw, zorder=1)
    ax.set_axis_off()
    return x, y


TOY = "((((A:1,B:1):1,(C:1,D:1):1):1,((E:1,F:1):1,G:2):1):1,((H:1.5,I:1.5):1.5,J:3):1);"


# ---------------------------------------------------------------------------------------
# Figure 1: method overview
# ---------------------------------------------------------------------------------------

def figure1():
    tree, names = from_newick(TOY)
    fig = plt.figure(figsize=(7.2, 5.2))
    gs = fig.add_gridspec(2, 3, hspace=0.75, wspace=0.45)

    # a: trait and independent origins
    trait = np.isin(names, ["A", "B", "E", "H", "I"])
    fg = foreground_branches(tree, trait)
    ax = fig.add_subplot(gs[0, 0])
    x, y = draw_tree(ax, tree, {b: C1 for b in np.nonzero(fg)[0]})
    for t in range(tree.n_tips):
        ax.text(x[t] + 0.15, y[t], names[t], va="center", fontsize=7,
                color=C1 if trait[t] else INK2, fontweight="bold" if trait[t] else "normal")
    ax.set_xlim(-0.2, x.max() + 0.9)
    ax.invert_yaxis()
    panel(ax, "a", "Trait: 3 origins")

    # b: data layers per gene
    ax = fig.add_subplot(gs[0, 1])
    rng = np.random.default_rng(3)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_axis_off()
    labels = ["Nucleotide rate", "Protein rate", "Selection (protein − nucleotide)", "Gene loss (TOGA)"]
    for i, lab in enumerate(labels):
        y0 = 8.4 - i * 2.4
        vals = rng.normal(size=12) if i < 3 else (rng.random(12) < 0.25).astype(float)
        cmap = "Greys" if i < 3 else ListedColormap(["#ffffff", C2])
        ax.imshow(vals[None], extent=(0.2, 6.2, y0 - 0.45, y0 + 0.45), cmap=cmap, aspect="auto",
                  vmin=-2.5 if i < 3 else 0, vmax=2.5 if i < 3 else 1)
        ax.add_patch(Rectangle((0.2, y0 - 0.45), 6.0, 0.9, fill=False, ec=INK2, lw=0.6))
        ax.text(0.2, y0 + 0.6, lab, fontsize=7, color=INK)
    ax.text(6.5, 6.0, "per branch\n(log length,\nmean-centred)", fontsize=6.5, color=INK2, va="center")
    ax.text(6.5, 1.2, "per species\n(lost / intact)", fontsize=6.5, color=INK2, va="center")
    panel(ax, "b", "Layers per gene")

    # c: genome-learned covariance
    ax = fig.add_subplot(gs[0, 2])
    B = 12
    K = np.exp(-np.abs(np.subtract.outer(np.arange(B), np.arange(B))) / 2.5)
    K += 0.3 * np.diag(rng.random(B))
    ax.imshow(K, cmap="Blues", vmin=0, vmax=1.2)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(INK2)
    ax.set_xlabel("branches (or species)", fontsize=7)
    ax.text(0.5, -0.2, "from ~15,000 genes", transform=ax.transAxes, ha="center", va="top",
            fontsize=6.5, color=INK2)
    panel(ax, "c", "Genome-learned null Σ")

    # d: leave-one-origin-out
    origins_tips = [["A", "B"], ["E"], ["H", "I"]]
    sub = gs[1, 0:2].subgridspec(1, 3, wspace=0.1)
    for k in range(3):
        ax = fig.add_subplot(sub[0, k])
        keep_trait = np.isin(names, sum([o for j, o in enumerate(origins_tips) if j != k], []))
        drop = np.isin(names, origins_tips[k])
        fgk = foreground_branches(tree, keep_trait)
        fgd = foreground_branches(tree, drop)
        col = {b: C1 for b in np.nonzero(fgk)[0]}
        col.update({b: "#d0cfca" for b in np.nonzero(fgd)[0]})
        x, y = draw_tree(ax, tree, col, lw=1.3)
        for t in np.nonzero(drop)[0]:
            ax.text(x[t] + 0.15, y[t], "×", va="center", ha="left", fontsize=8, color=MUTED)
        ax.set_xlim(-0.2, x.max() + 0.8)
        ax.invert_yaxis()
        ax.text(0.5, -0.06, f"p{k + 1}: origin {k + 1} removed",
                transform=ax.transAxes, ha="center", fontsize=6.5, color=INK2)
        if k == 0:
            panel(ax, "d", "Leave one origin out: p = max(p1, p2, p3)")

    # e: calibration
    ax = fig.add_subplot(gs[1, 2])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_axis_off()
    boxes = [(8.0, "Matched control trait sets\non real genomes\n(every call is false)", C3),
             (5.0, "Simulated null for gene loss\n(loss rate × gene fragility ×\nspecies error rate)", C2),
             (1.8, "FDR (Benjamini–Hochberg)\nper layer, q = 0.1", C1)]
    for yy, text, c in boxes:
        ax.add_patch(FancyBboxPatch((0.3, yy - 1.1), 9.4, 2.2, boxstyle="round,pad=0.05,rounding_size=0.3",
                                    fc="white", ec=c, lw=1.4))
        ax.text(5.0, yy, text, ha="center", va="center", fontsize=6.2, color=INK)
    panel(ax, "e", "Calibration")
    save(fig, "Figure1")


# ---------------------------------------------------------------------------------------
# Figure 2: rate layers on AliSim data
# ---------------------------------------------------------------------------------------

METHODS_RATE = [("naive", "Naive (RERconverge-style)", C1), ("permulation", "Control-lineage permutation", C2),
                ("naive_fisher", "Naive, Fisher over layers", C3), ("brownian_lmm", "Brownian phylo-LMM", C4),
                ("gls", "Learned-covariance GLS", C5), ("final", "GLS + leave-one-origin-out", C6)]


# kinds of convergent change, all for the final method (its colour), told apart by line and marker
KINDS = (("power_both", "Selection + rate", C6, "-", "o"), ("power_omega", "Selection (ω)", C6, "--", "s"),
         ("power_rate", "Rate only", C6, ":", "^"))


def figure2():
    df = pd.read_csv(RES / "final_benchmark.csv")
    og = pd.read_csv(RES / "origins_power.csv")
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.9), gridspec_kw=dict(hspace=0.6, wspace=0.42))
    conf = sorted(df.confounding.unique())
    x = np.arange(len(conf))
    w = 0.13

    ax = axes[0, 0]
    null = df[df.effect == "weak"].groupby(["confounding", "method"]).false_calls_null.mean()
    for i, (m, lab, c) in enumerate(METHODS_RATE):
        v = np.array([null.get((cf, m), np.nan) for cf in conf])
        ax.scatter(x + (i - 2.5) * w, v + 1, color=c, s=14, zorder=3, label=lab, edgecolor="white", lw=0.5)
    ax.set_yscale("log")
    ax.set_xticks(x, [f"{c:g}" for c in conf])
    ax.set_xlabel("Rate-jump intensity")
    ax.set_ylabel("False calls + 1")
    grid(ax)
    panel(ax, "a", "No convergence")

    ax = axes[0, 1]
    fdr = df.groupby(["confounding", "method"]).fdr.mean()
    for i, (m, lab, c) in enumerate(METHODS_RATE):
        v = np.array([fdr.get((cf, m), np.nan) for cf in conf])
        ax.scatter(x + (i - 2.5) * w, v, color=c, s=14, zorder=3, edgecolor="white", lw=0.5)
    ax.axhline(0.1, color=INK2, lw=0.8, ls="--")
    ax.set_xticks(x, [f"{c:g}" for c in conf])
    ax.set_ylim(-0.03, 1)
    ax.set_xlabel("Rate-jump intensity")
    ax.set_ylabel("Realised FDR (q = 0.1)")
    grid(ax)
    panel(ax, "b", "2% of genes convergent")

    ax = axes[0, 2]
    fin = df[df.method == "final"]
    eff = ["weak", "moderate", "strong"]
    for col, lab, c, ls, mk in KINDS:
        v = [fin[fin.effect == e][col].mean() for e in eff]
        ax.plot(range(3), v, color=c, lw=1.8, ls=ls, marker=mk, ms=4, label=lab)
    ax.set_xticks(range(3), eff)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Power (final test)")
    ax.set_xlabel("Effect size")
    ax.legend(loc="lower right")
    grid(ax)
    panel(ax, "c", "Power by kind of change")

    ax = axes[1, 0]
    g = fin.groupby("effect").signature_accuracy
    m, lo, hi = (g.mean().reindex(eff), g.min().reindex(eff), g.max().reindex(eff))
    ax.errorbar(range(3), m, yerr=[m - lo, hi - m], color=C6, lw=1.8, marker="o", ms=4, capsize=3)
    ax.set_xticks(range(3), eff)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Exact signature accuracy")
    ax.set_xlabel("Effect size")
    grid(ax)
    panel(ax, "d", "Signatures of called genes")

    ax = axes[1, 1]
    fl = og[og.method == "gls_looo_acat"].groupby("origins")
    for col, lab, c, ls, mk in KINDS:
        ax.plot(fl[col].mean().index, fl[col].mean().values, color=c, lw=1.8, ls=ls, marker=mk, ms=4, label=lab)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Independent origins")
    ax.set_ylabel("Power (final test)")
    grid(ax)
    panel(ax, "e", "Power by number of origins")

    ax = axes[1, 2]
    for m, lab, c in (("naive:rate_nucleotide", "Naive", C1), ("gls_acat", "Learned-covariance GLS", C5),
                      ("gls_looo_acat", "GLS + leave-one-origin-out", C6)):
        s = og[og.method == m].groupby("origins").fdr.mean()
        ax.plot(s.index, s.values, color=c, lw=1.8, marker="o", ms=4, label=lab)
    ax.axhline(0.1, color=INK2, lw=0.8, ls="--")
    ax.set_ylim(-0.03, 1)
    ax.set_xlabel("Independent origins")
    ax.set_ylabel("Realised FDR (q = 0.1)")
    grid(ax)
    panel(ax, "f", "False discoveries by origins")

    handles = [plt.Line2D([], [], marker="o", ls="", color=c, ms=5, label=lab) for _, lab, c in METHODS_RATE]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    save(fig, "Figure2")


# ---------------------------------------------------------------------------------------
# Figure 3: gene-loss layer
# ---------------------------------------------------------------------------------------

LOSS_METHODS = [("simulation", "This method (simulated null)", C1), ("gls", "Same statistic, GLS p-value", C2),
                ("fisher", "Fisher exact, no phylogeny", C3)]
# loss-null development on matched control trait sets of real genomes (README, "Gene-loss layer"
# and "Simulation null" sections; final row pooled over the five traits of the paper dataset)
LOSS_NULLS = [("GLS t-test", 60, 5.1), ("Control sets, 200 available", 20, 0.25),
              ("Control sets, 142 available", 10, 0.15), ("Simulated, species error", 5, 0.05),
              ("Simulated, species \u00d7 gene error", 8, 0.10)]


def figure3():
    d = pd.read_csv(RES / "loss_benchmark.csv")
    sig = d[d.penetrance > 0]
    origins = sorted(d.origins.unique())
    fig = plt.figure(figsize=(7.2, 6.2))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.9], hspace=0.65, wspace=0.18)
    letters = iter("abcdef")
    for row, (col, ylab, ylim) in enumerate((("power", "Power", (0, 1)), ("fdp", "False discovery proportion", (0, 0.6)))):
        for j, k in enumerate(origins):
            ax = fig.add_subplot(gs[row, j])
            for m, lab, c in LOSS_METHODS:
                g = sig[(sig.origins == k) & (sig.method == m)].groupby("penetrance")[col]
                ax.plot(g.mean().index, g.mean().values, color=c, lw=1.8, marker="o", ms=4,
                        markeredgecolor="white", markeredgewidth=0.8, label=lab)
                ax.vlines(g.mean().index, g.min().values, g.max().values, color=c, lw=0.8)
            if col == "fdp":
                ax.axhline(0.1, color=INK2, lw=0.8, ls="--")
                ax.set_xlabel("Penetrance")
            ax.set_ylim(*ylim)
            ax.set_xticks([0.5, 0.75, 1.0])
            if j == 0:
                ax.set_ylabel(ylab)
            else:
                ax.set_yticklabels([])
            grid(ax)
            panel(ax, next(letters), f"{k} origins")
    fig.legend(handles=[plt.Line2D([], [], color=c, lw=1.8, marker="o", ms=4, label=lab) for _, lab, c in LOSS_METHODS],
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.975))

    ax = fig.add_subplot(gs[2, :])
    yy = np.arange(len(LOSS_NULLS))
    vals = [v for _, v, _ in LOSS_NULLS]
    ax.hlines(yy, 0, vals, color=GRID, lw=3, zorder=1)
    ax.scatter(vals, yy, s=40, color=[MUTED] * 4 + [C1], edgecolor="white", lw=1, zorder=3)   # final = this method
    for v, yv, (_, _, mean) in zip(vals, yy, LOSS_NULLS):
        ax.text(max(v, 10) + 1.5, yv, f"{v}%  ({mean:g} false calls per set)", va="center", fontsize=6.8, color=INK2)
    ax.axvline(10, color=INK2, lw=0.8, ls="--")
    ax.set_yticks(yy, [n for n, _, _ in LOSS_NULLS])
    ax.invert_yaxis()
    ax.set_xlim(0, 85)
    ax.set_xlabel("Matched control trait sets with any loss call (%), real genomes")
    grid(ax, "x")
    panel(ax, "g", "Calibration of loss nulls on real data")
    save(fig, "Figure3")


# ---------------------------------------------------------------------------------------
# Figure 4: genome-wide results
# ---------------------------------------------------------------------------------------

TRAITS = [("marine", "Marine"), ("hypoxia_subterranean", "Hypoxia, subterranean"), ("hypoxia", "Hypoxia, combined"),
          ("hypoxia_altitude", "Hypoxia, high altitude"), ("echolocation", "Echolocation")]
LAYERS = [("rate", "Rate", C1), ("loss", "Gene loss", C2), ("final", "Combined", C3)]


def loss_panel(ax, trait_key, genes, title, letter, tree, names, status, trait_species, colour):
    order = names                                   # Newick order = tip order of the drawn tree
    M = status.reindex(index=genes, columns=order).fillna("M")
    code = np.where(M.isin(["L", "UL"]), 2, np.where(M.isin(["I", "PI"]), 1, 0))
    cmap = ListedColormap(["#ffffff", "#e8e7e3", INK])
    ax.imshow(code, aspect="auto", cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    is_t = np.isin(order, trait_species)
    strip = blended_transform_factory(ax.transData, ax.transAxes)   # x in species, y in axes fraction
    for t in np.nonzero(is_t)[0]:
        ax.add_patch(Rectangle((t - 0.5, 1.012), 1, 0.035, color=colour, clip_on=False, lw=0, transform=strip))
    ax.set_yticks(range(len(genes)), genes, fontsize=5.6)
    ax.set_xticks([])
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_xlabel(f"{len(order)} species in the order of the Zoonomia tree", fontsize=6.5)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(f"{letter}  {title}", loc="left", fontsize=9, color=INK, pad=14)


def figure4():
    rows = []
    for key, label in TRAITS:
        s = json.loads((RES / f"{key}_toga" / "summary.json").read_text())
        for lay, _, _ in LAYERS:
            rows.append(dict(trait=label, layer=lay, calls=s[f"calls_{lay}"],
                             any_call=100 * s[f"control_sets_with_any_call_{lay}"], origins=s["origins"]))
    d = pd.DataFrame(rows)
    tree, names = from_newick((RES / "toga" / "master_paper.nwk").read_text())   # pruned Zoonomia tree
    status = pd.read_csv(RES / "toga" / "loss_status_paper.tsv.gz", sep="\t", index_col=0)
    mar = pd.read_csv(RES / "marine_toga" / "marine_genes.csv")
    sub = pd.read_csv(RES / "hypoxia_subterranean_toga" / "hypoxia_subterranean_genes.csv")
    mar_genes = mar[mar.call_loss == True].sort_values("p_loss").symbol.tolist()  # noqa: E712
    sub_genes = sub[sub.call_loss == True].sort_values("p_loss").symbol.tolist()  # noqa: E712

    fig = plt.figure(figsize=(7.2, 10.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.3, 3.4, 1.2], hspace=0.42, wspace=0.75)
    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(TRAITS))
    h = 0.26
    for i, (lay, name, col) in enumerate(LAYERS):
        v = d[d.layer == lay].calls.to_numpy()
        ax.barh(y + (i - 1) * h, v, height=h - 0.04, color=col, label=name)
        for yy, vv in zip(y + (i - 1) * h, v):
            if vv:
                ax.text(vv + 0.8, yy, str(vv), va="center", fontsize=6, color=INK2)
    origins = d.drop_duplicates("trait").origins.tolist()
    ax.set_yticks(y, [f"{t} ({o})" for (_, t), o in zip(TRAITS, origins)], fontsize=7)
    ax.set_ylim(len(TRAITS) - 0.5, -0.95)
    ax.set_xlabel("Genes called (q = 0.1)")
    ax.legend(handles=[Rectangle((0, 0), 1, 1, color=c, label=n + (" (no calls)" if lay == "rate" else ""))
                       for lay, n, c in LAYERS], loc="lower right")
    grid(ax, "x")
    panel(ax, "a", "Convergent genes (origins)")

    ax = fig.add_subplot(gs[0, 1])
    for i, (lay, name, col) in enumerate(LAYERS):
        v = d[d.layer == lay].any_call.to_numpy()
        ax.scatter(v, y + (i - 1) * 0.22, s=22, color=col, edgecolor="white", lw=1, zorder=3, label=name)
    ax.axvline(10, color=INK2, lw=0.8, ls="--")
    ax.set_yticks(y, [""] * len(y))
    ax.set_ylim(len(TRAITS) - 0.5, -0.95)
    ax.text(10.5, -0.72, "nominal 10%", fontsize=6, color=INK2, va="center")
    ax.set_xlim(-1.5, 25)
    ax.set_xlabel("Control trait sets with any call (%)")
    grid(ax, "x")
    panel(ax, "b", "Calibration (20 control sets each)")

    mar_species = (RES / "marine_toga" / "trait_species.txt").read_text().split()
    sub_species = (RES / "hypoxia_subterranean_toga" / "trait_species.txt").read_text().split()
    loss_panel(fig.add_subplot(gs[1, :]), "marine", mar_genes, f"Marine mammals: {len(mar_genes)} convergent losses",
               "c", tree, names, status, mar_species, C5)
    loss_panel(fig.add_subplot(gs[2, :]), "sub", sub_genes,
               f"Subterranean mammals: {len(sub_genes)} convergent losses", "d", tree, names, status, sub_species, C5)
    handles = [Rectangle((0, 0), 1, 1, color=INK, label="lost (TOGA L or UL)"),
               Rectangle((0, 0), 1, 1, color="#e8e7e3", label="intact (I or PI)"),
               Rectangle((0, 0), 1, 1, fc="white", ec=MUTED, lw=0.5, label="missing"),
               Rectangle((0, 0), 1, 1, color=C5, label="trait species")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.06))
    save(fig, "Figure4")


# ---------------------------------------------------------------------------------------
# Figure 5: timing of convergent losses and partitioning of correlated traits
# ---------------------------------------------------------------------------------------

ORIGIN_NAMES = {"Eubalaena": "Cetaceans", "Zalophus": "Pinnipeds", "Trichechus": "Manatee*",
                "Heterocephalus": "African mole-rats", "Ctenomys": "Tuco-tuco*", "Nannospalax": "Blind mole-rat*",
                "Ellobius": "Mole vole*", "Condylura": "Moles", "Chrysochloris": "Golden mole*"}
TIMING = [("before origin", "Before origin (shared with sister)", "#6250d6"),
          ("at origin", "On the origin's stem branch", C1), ("within origin", "Later, within the origin", C2),
          ("intact", "Intact", "#e8e7e3"), ("no data", "No data", "#ffffff")]


def timing_panel(ax, t, trait, genes, letter, title):
    d = t[t.trait == trait]
    origins = list(dict.fromkeys(d.origin))
    code = {c: i for i, (c, _, _) in enumerate(TIMING)}
    M = np.array([[code[d[(d.gene == g) & (d.origin == o)].category.iloc[0]] for o in origins] for g in genes])
    ax.imshow(M, aspect="auto", cmap=ListedColormap([c for _, _, c in TIMING]), vmin=0, vmax=len(TIMING) - 1,
              interpolation="nearest")
    ax.set_yticks(range(len(genes)), genes, fontsize=5.4)
    ax.set_xticks(range(len(origins)), [ORIGIN_NAMES[o.split("+")[0]] for o in origins], rotation=40, ha="right",
                  fontsize=6.5)
    ax.set_xticks(np.arange(len(origins) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(genes) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", lw=0.6)
    ax.tick_params(which="both", length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    panel(ax, letter, title)


def joint_panel(ax, df, xa, xb, called, controls, letter, title, xlabel, ylabel):
    ax.scatter(df[xa], df[xb], s=3, color="#c9c8c3", lw=0, rasterized=True, label="all genes")
    c = df[df.gene.isin(called)]
    ax.scatter(c[xa], c[xb], s=14, color=C1, edgecolor="white", lw=0.5, zorder=3, label="convergent losses")
    k = df[df.gene.isin(controls)]
    ax.scatter(k[xa], k[xb], s=16, color=C2, marker="D", edgecolor="white", lw=0.5, zorder=4, label="genes of interest")
    for i, (_, r) in enumerate(k.sort_values(xa).iterrows()):
        dx, ha = ((-4, "right") if i % 2 == 0 else (4, "left"))
        ax.annotate(r.gene, (r[xa], r[xb]), fontsize=5.5, color=INK2, xytext=(dx, 3), textcoords="offset points",
                    ha=ha)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.35, 1.1)
    ax.set_ylim(-0.35, 1.1)
    ax.set_aspect("equal")
    grid(ax, "both")
    panel(ax, letter, title)


def figure5():
    t = pd.read_csv(RES / "extra" / "loss_timing.csv")
    hyp = pd.read_csv(RES / "extra" / "joint_subterranean_altitude.csv")
    mar = pd.read_csv(RES / "extra" / "joint_marine_semiaquatic.csv")
    mg = list(dict.fromkeys(t[t.trait == "marine"].gene))
    sg = list(dict.fromkeys(t[t.trait == "hypoxia_subterranean"].gene))
    fig = plt.figure(figsize=(7.2, 9.6))
    gs = fig.add_gridspec(3, 3, height_ratios=[2.2, 0.32, 1.35], width_ratios=[1, 1.25, 0.9], hspace=0.3,
                          wspace=0.75)
    timing_panel(fig.add_subplot(gs[0, 0]), t, "marine", mg, "a", "Marine losses")
    ax = fig.add_subplot(gs[0, 1])
    timing_panel(ax, t, "hypoxia_subterranean", sg, "b", "Subterranean losses")
    ax.set_anchor("N")
    ax = fig.add_subplot(gs[0, 2])
    tot = t[t.category != "no data"].groupby(["trait", "category"]).size().unstack(fill_value=0)
    tot = tot.div(tot.sum(1), axis=0)
    bottom = np.zeros(len(tot))
    for cat, lab, col in TIMING[:4]:
        v = tot.get(cat, pd.Series(0, index=tot.index)).to_numpy()
        ax.bar(range(len(tot)), v, bottom=bottom, color=col, width=0.7, edgecolor="white", lw=1, label=lab)
        bottom += v
    ax.set_xticks(range(len(tot)), ["Marine" if i == "marine" else "Subter-\nranean" for i in tot.index], fontsize=7)
    ax.set_ylabel("Share of gene \u00d7 origin pairs")
    ax.set_ylim(0, 1)
    ax.set_anchor("N")
    panel(ax, "c", "Timing")
    lg = fig.add_subplot(gs[1, :])
    lg.set_axis_off()
    lg.legend(handles=[Rectangle((0, 0), 1, 1, fc=c, ec=MUTED if c == "#ffffff" else c, lw=0.5, label=l)
                       for _, l, c in TIMING], loc="lower center", ncol=3)
    bottom_row = gs[2, :].subgridspec(1, 2, wspace=0.35)
    joint_panel(fig.add_subplot(bottom_row[0, 0]), hyp, "beta_subterranean", "beta_altitude", sg,
                ["EPAS1", "EGLN1", "HIF1A", "HBB", "NGB", "CYGB"], "d", "Subterranean vs high altitude",
                "Partial effect: subterranean", "Partial effect: high altitude")
    ax = fig.add_subplot(bottom_row[0, 1])
    joint_panel(ax, mar, "beta_marine", "beta_semiaquatic", mg, ["PON1", "MMP12"], "e",
                "Marine vs semi-aquatic", "Partial effect: marine", "Partial effect: semi-aquatic")
    ax.legend(loc="upper right", fontsize=6, markerscale=1.2)
    save(fig, "Figure5")


# ---------------------------------------------------------------------------------------
# Supplementary Figure 2: loss model and data quality
# ---------------------------------------------------------------------------------------

def supp2():
    q = pd.read_csv(RES / "extra" / "species_quality.csv").dropna(subset=["contig_n50"])
    ag = pd.read_csv(RES / "extra" / "adequacy_losses_per_gene.csv")
    asp = pd.read_csv(RES / "extra" / "adequacy_losses_per_species.csv")
    mp = pd.read_csv(RES / "extra" / "loss_model_parameters.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), gridspec_kw=dict(hspace=0.55, wspace=0.4))
    ax = axes[0, 0]
    ax.scatter(q.contig_n50, 100 * q.false_loss_rate, s=12, color=C1, edgecolor="white", lw=0.5)
    ax.set_xscale("log")
    from scipy import stats as st_
    r = st_.spearmanr(np.log10(q.contig_n50), q.false_loss_rate)
    ax.text(0.97, 0.95, f"Spearman \u03c1 = {r.statistic:.2f}, p = {r.pvalue:.2f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.5, color=INK2)
    ax.set_xlabel("Contig N50 (bp)")
    ax.set_ylabel("Species false-loss rate (%)")
    grid(ax, "both")
    panel(ax, "a", "Error rate vs assembly contiguity")

    ax = axes[0, 1]
    vals, counts = np.unique(mp.fragility, return_counts=True)
    ax.bar(range(len(vals)), counts / counts.sum(), color=C1, width=0.7)
    ax.set_xticks(range(len(vals)), [f"{v:g}" for v in vals])
    ax.set_xlabel("Gene fragility \u03ba (multiplier of species error rate)")
    ax.set_ylabel("Share of genes")
    grid(ax)
    panel(ax, "b", "Fitted gene fragility")

    ax = axes[1, 0]
    x = np.arange(len(ag))
    labels = [f"{a}" if a == b else f"{a}\u2013{b}" for a, b in zip(ag.bin_low, ag.bin_high)]
    ax.bar(x - 0.2, ag.observed, width=0.4, color=INK2, label="observed")
    ax.bar(x + 0.2, ag.sim_mean, width=0.4, color=C1, label="simulated (20 genomes)")
    ax.vlines(x + 0.2, ag.sim_low, ag.sim_high, color=INK, lw=0.8)
    ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=6.5)
    ax.set_xlabel("Species with the gene lost")
    ax.set_ylabel("Share of testable genes")
    ax.legend(loc="upper right")
    grid(ax)
    panel(ax, "c", "Losses per gene")

    ax = axes[1, 1]
    ax.errorbar(100 * asp.observed, 100 * asp.sim_mean,
                yerr=[100 * (asp.sim_mean - asp.sim_low), 100 * (asp.sim_high - asp.sim_mean)],
                fmt="o", ms=3, color=C1, ecolor=GRID, elinewidth=0.8)
    lim = 100 * max(asp.observed.max(), asp.sim_high.max()) * 1.05
    ax.plot([0, lim], [0, lim], color=INK2, lw=0.8, ls="--")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Observed share of genes lost (%)")
    ax.set_ylabel("Simulated share (%)")
    grid(ax, "both")
    panel(ax, "d", "Losses per species")
    save(fig, "SupplementaryFigure2")


# ---------------------------------------------------------------------------------------
# Supplementary Figure 1: RERconverge
# ---------------------------------------------------------------------------------------

def supp1():
    r = pd.read_csv(RES / "rerconverge_compare.csv")
    meth = [("final", "This method (final)", C6), ("RERconverge_protein", "RERconverge, protein", C1),
            ("RERconverge_nucleotide", "RERconverge, nucleotide", C2),
            ("RERconverge_protein_permulation", "RERconverge + permulations", "#6250d6")]   # violet, slot 7
    conf = sorted(r.confounding.unique())
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw=dict(wspace=0.45))
    x = np.arange(len(conf))
    w = 0.18
    for ax, (eff, col, ylab, letter, title) in zip(axes, (
            ("none", "false_calls", "False calls", "a", "No convergence"),
            ("moderate", "fdr", "Realised FDR (q = 0.1)", "b", "2% of genes convergent"),
            ("moderate", "power", "Power", "c", "2% of genes convergent"))):
        s = r[r.effect == eff].groupby(["confounding", "method"])[col].mean()
        for i, (m, lab, c) in enumerate(meth):
            v = [s.get((cf, m), np.nan) for cf in conf]
            ax.bar(x + (i - 1.5) * w, v, width=w - 0.02, color=c, label=lab)
            for xx, vv in zip(x + (i - 1.5) * w, v):
                if vv == 0:
                    ax.text(xx, 0, "0", ha="center", va="bottom", fontsize=5.5, color=INK2)
        if col == "fdr":
            ax.axhline(0.1, color=INK2, lw=0.8, ls="--")
        ax.set_xticks(x, [f"{c:g}" for c in conf])
        ax.set_xlabel("Rate-jump intensity")
        ax.set_ylabel(ylab)
        grid(ax)
        panel(ax, letter, title)
    fig.legend(handles=[Rectangle((0, 0), 1, 1, color=c, label=lab) for _, lab, c in meth],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.2))
    save(fig, "SupplementaryFigure1")


if __name__ == "__main__":
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    supp1()
    supp2()
