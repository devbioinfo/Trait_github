"""Numbered manuscript figures (PNG at 300 dpi and PDF) in results/figures/.

Figure 1  the 115 species, the four traits and their independent origins (circular tree)
Figure 2  method overview (schematic)
Figure 3  rate layers on simulated codon alignments (calibration, power, signatures, origins)
Figure 4  gene-loss layer: simulation benchmark and calibration of loss nulls on real genomes
Figure 5  genome-wide results for three traits: calls, calibration, and the convergent losses
Figure 6  timing of convergent losses and partitioning of correlated traits
Figure 7  marine losses in detail: lineage funnels, PON1 mutation map, exon synteny
Figure 8  codon alignments of two losses, genome-wide volcano plots, functional map
Figure 9  tissue, pathway and GO enrichment of the convergent losses
Figure 10 pathway maps: taste and smell in marine mammals, the eye in subterranean mammals
Supplementary Figure 2  loss model: data quality and goodness of fit
Supplementary Figure 1  comparison with RERconverge
Supplementary Figure 3  number of origins: subterranean losses recovered from subsets of origins

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
from matplotlib.patches import FancyBboxPatch, PathPatch, Rectangle  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from mlconv.tree import foreground_branches, from_newick  # noqa: E402

RES = ROOT / "results"
OUT = RES / "figures"
OUT.mkdir(exist_ok=True)

# validated categorical palette (light), in fixed order; text in neutral inks
C1, C2, C3, C4, C5, C6 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
INK, INK2, MUTED, GRID, PALE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#f1f0ec"
# Nature Portfolio figure style: sans-serif (Arial metrics), text 5-7 pt, panel letters 8 pt bold
# lower case, figures at most 183 mm wide and 170 mm tall, editable text in the PDF
MM = 1 / 25.4
MAX_W, MAX_H = 183, 170                       # mm
MIN_PT = 5.0
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 6, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.labelsize": 6.5, "xtick.labelsize": 5.5, "ytick.labelsize": 5.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 7,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                     "legend.fontsize": 6, "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "mathtext.default": "regular"})


def panel(ax, letter, title="", pad=6):
    """Panel letter (8 pt bold, lower case) at the top left, with an optional 7 pt title beside it."""
    if title:
        ax.annotate(title, xy=(0, 1), xycoords="axes fraction", xytext=(10, pad), textcoords="offset points",
                    fontsize=7, color=INK, ha="left", va="baseline", annotation_clip=False)
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(0, pad), textcoords="offset points",
                fontsize=8, fontweight="bold", color=INK, ha="left", va="baseline", annotation_clip=False)


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def _extent(t, r):
    from matplotlib.text import Text
    return Text.get_window_extent(t, r)          # text only (an Annotation's own extent includes its arrow)


def _hidden_ticklabels(fig):
    """Tick labels that matplotlib keeps but does not draw (ticks outside the view limits)."""
    hidden = set()
    for ax in fig.axes:
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = min(lim), max(lim)
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                loc = tick.get_loc()
                if loc is None or not (lo - 1e-9 <= loc <= hi + 1e-9) or not tick.get_visible():
                    hidden.update({id(tick.label1), id(tick.label2)})
                if not tick.label2.get_visible() or not tick.label2On if hasattr(tick, "label2On") else False:
                    hidden.add(id(tick.label2))
    return hidden


def _text_box(t, r):
    """Oriented box of a text artist in display pixels: centre, half-sizes, angle (radians)."""
    bb = _extent(t, r)
    orig = t.get_rotation()
    rot = orig % 180
    if min(rot % 90, 90 - rot % 90) < 1e-6:
        return np.array([(bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2]), np.array([bb.width / 2, bb.height / 2]), 0.0
    t.set_rotation(0)
    b0 = _extent(t, r)
    t.set_rotation(orig)                     # restore exactly (the artist is saved afterwards)
    return np.array([(bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2]), np.array([b0.width / 2, b0.height / 2]), np.radians(rot)


def _boxes_overlap(a, b, tol=1.5):
    """Separating-axis test for two oriented rectangles, with a tolerance in pixels."""
    (ca, ha, ra), (cb, hb, rb) = a, b
    axes = [np.array([np.cos(r_), np.sin(r_)]) for r_ in (ra, rb)]
    axes += [np.array([-np.sin(r_), np.cos(r_)]) for r_ in (ra, rb)]
    ua = [np.array([np.cos(ra), np.sin(ra)]), np.array([-np.sin(ra), np.cos(ra)])]
    ub = [np.array([np.cos(rb), np.sin(rb)]), np.array([-np.sin(rb), np.cos(rb)])]
    d = cb - ca
    for n in axes:
        pa = ha[0] * abs(ua[0] @ n) + ha[1] * abs(ua[1] @ n)
        pb = hb[0] * abs(ub[0] @ n) + hb[1] * abs(ub[1] @ n)
        if abs(d @ n) > pa + pb - tol:
            return False
    return True


def check_figure(fig, name):
    """Report text below 5 pt, overlapping text (oriented boxes) and figures above 183 x 170 mm."""
    from matplotlib.text import Text
    import warnings
    fig.set_dpi(300)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig.canvas.draw()
    r = fig.canvas.get_renderer()
    missing = sorted({str(w.message).split(" missing")[0] for w in caught if "Glyph" in str(w.message)})
    hidden = _hidden_ticklabels(fig)
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip() and t.get_alpha() != 0
             and id(t) not in hidden]
    problems, boxes = [f"missing glyph: {g}" for g in missing], []
    for t in texts:
        if t.get_fontsize() < MIN_PT - 1e-6:
            problems.append(f"small text {t.get_fontsize():.1f} pt: {t.get_text()[:30]!r}")
        bb = _extent(t, r)
        if bb.width > 0 and bb.height > 0:
            boxes.append((bb, _text_box(t, r), t))
    for i in range(len(boxes)):
        a, oa, ta = boxes[i]
        for j in range(i + 1, len(boxes)):
            b, ob, tb = boxes[j]
            if min(a.x1, b.x1) - max(a.x0, b.x0) <= 1.5 or min(a.y1, b.y1) - max(a.y0, b.y0) <= 1.5:
                continue
            if _boxes_overlap(oa, ob):
                problems.append(f"overlap: {ta.get_text()[:25]!r} / {tb.get_text()[:25]!r}")
    bb = fig.get_tightbbox(r)
    wmm, hmm = bb.width * 25.4, bb.height * 25.4
    if wmm > MAX_W + 0.5 or hmm > MAX_H + 0.5:
        problems.append(f"size {wmm:.0f} x {hmm:.0f} mm (max {MAX_W} x {MAX_H})")
    print(f"check {name}: {wmm:.0f} x {hmm:.0f} mm, {len(problems)} problem(s)")
    for p_ in problems[:40]:
        print("   ", p_)
    return problems


def save(fig, name):
    check_figure(fig, name)
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
# Figure 1: the 115 species, the four traits and their independent origins
# ---------------------------------------------------------------------------------------

ORDERS = [("Chiroptera", "Bats"), ("Cetacea", "Whales and dolphins"), ("Hippopotamidae", "Hippo"),
          ("Tylopoda", "Camelids"), ("Artiodactyla", "Ruminants, pigs"),
          ("Carnivora", "Carnivores"), ("Perissodactyla", "Perissodactyls"), ("Pholidota", "Pangolins"),
          ("Eulipotyphla", "Moles, hedgehog"), ("Primates", "Primates"), ("Rodentia", "Rodents"),
          ("Lagomorpha", "Rabbits, pika"), ("Scandentia", "Treeshrews"), ("Dermoptera", "Colugo"),
          ("Afrotheria", "Afrotherians"), ("Xenarthra", "Xenarthrans"), ("Metatheria", "Marsupials"),
          ("Prototheria", "Monotreme")]
SHORT_NAMES = {"Homo_sapiens": "human", "Nannospalax_galili": "blind mole-rat",
               "Choloepus_hoffmanni": "two-toed sloth", "Choloepus_didactylus": "southern two-toed sloth",
               "Peromyscus_maniculatus_bairdii": "deer mouse", "Mesoplodon_bidens": "Sowerby's beaked whale",
               "Erinaceus_europaeus": "European hedgehog"}
TREE_TRAITS = [("echolocation", "Echolocation", C2), ("marine", "Marine", C1),
               ("hypoxia_subterranean", "Subterranean", C3), ("hypoxia_altitude", "High altitude", C4)]


def figure_tree():
    import convergence_toga as CT
    from mlconv.tree import origin_nodes as onodes
    tree, names = from_newick((RES / "toga" / "master_paper.nwk").read_text())
    cov = pd.read_csv(RES / "toga" / "coverage_paper.tsv", sep="\t").drop_duplicates("species").set_index("species")
    lineage = cov.lineage.reindex(names).fillna("")
    lineage.iloc[names.index("Homo_sapiens")] = "Eutheria; Euarchontoglires; Primates; Haplorrhini"
    common = pd.read_csv(RES / "toga" / "assemblies_paper.tsv", sep="\t").set_index("species")["Common name"]
    n = tree.n_tips
    _, y = tree_xy(tree)
    # cladogram: radius from the number of nodes between a node and its deepest tip, tips at 1
    height = np.zeros(tree.n_nodes)
    for v in range(tree.n_nodes - 1):
        height[tree.parent[v]] = max(height[tree.parent[v]], height[v] + 1)
    x = 1 - height / height.max()
    R = 1.0
    span = 2 * np.pi * 0.965
    theta = lambda v: np.pi / 2 - y[v] / (n - 1) * span          # noqa: E731  clockwise from the top
    fig = plt.figure(figsize=(183 * MM, 170 * MM))
    ax = fig.add_axes([0.06, 0.06, 0.88, 0.93], projection="polar")
    ax.set_axis_off()
    for v in range(tree.n_nodes - 1):
        p = tree.parent[v]
        ax.plot([theta(v), theta(v)], [x[p] * R, x[v] * R], color=MUTED, lw=0.7, solid_capstyle="butt")
    for v in range(n, tree.n_nodes):
        ch = np.nonzero(tree.parent == v)[0]
        a = np.linspace(min(theta(c) for c in ch), max(theta(c) for c in ch), 30)
        ax.plot(a, np.full_like(a, x[v] * R), color=MUTED, lw=0.7)
    # trait rings and origin markers
    width = span / (n - 1)
    ring0, ring_w = 1.04, 0.055
    D = tree.tip_sets()
    for k, (key, label, col) in enumerate(TREE_TRAITS):
        tr = np.array([CT.TRAITS[key][0](s_, l) for s_, l in zip(names, lineage)])
        r0 = ring0 + k * (ring_w + 0.008)
        th = np.array([theta(t) for t in range(n)])
        ax.bar(th, ring_w, width=width * 0.98, bottom=r0, color=np.where(tr, col, "#f1f0ec"), lw=0)
        for o in onodes(tree, tr):
            p = tree.parent[o]
            ax.scatter([theta(o)], [(x[p] + x[o]) / 2 * R], s=34, color=col, edgecolor="white", lw=0.8, zorder=5)
    # order arcs and species names
    group = []
    for l in lineage:
        group.append(next((lab for key, lab in ORDERS if key in l), ""))
    rl = ring0 + 4 * (ring_w + 0.008) + 0.02
    for t in range(n):
        ang = theta(t)
        deg = np.degrees(ang)
        flip = 90 < (deg % 360) < 270
        label = SHORT_NAMES.get(names[t], common.get(names[t], names[t].replace("_", " ")))
        label = label if len(label) <= 20 else label[:19].rstrip() + "\u2026"
        ax.text(ang, rl, label, fontsize=5, color=INK2,
                rotation=deg + (180 if flip else 0), rotation_mode="anchor", ha="right" if flip else "left",
                va="center")
    start = 0
    ro = rl + 0.5
    for t in range(1, n + 1):
        if t == n or group[t] != group[start]:
            if group[start] and t - start >= 1:
                a = np.linspace(theta(start) + width * 0.4, theta(t - 1) - width * 0.4, 40)
                ax.plot(a, np.full_like(a, ro), color=INK, lw=1.2)
                mid = (theta(start) + theta(t - 1)) / 2
                deg = np.degrees(mid) % 360
                flip = 180 < deg < 360
                ax.text(mid, ro + 0.03, group[start] if t - start >= 3 else "", fontsize=6, color=INK, ha="center",
                        va="top" if flip else "bottom", rotation=deg - 90 + (180 if flip else 0),
                        rotation_mode="anchor")
            start = t
    ax.set_ylim(0, ro + 0.24)
    th = np.array([theta(t) for t in range(n)])
    place_silhouettes(fig, ax, lineage, group, th, ro + 0.155, 0.03, names)
    handles = [plt.Line2D([], [], marker="s", ls="", color=c, ms=7, label=f"{lab} ({len(onodes(tree, np.array([CT.TRAITS[k][0](s_, l) for s_, l in zip(names, lineage)])))} origins)")
               for k, lab, c in TREE_TRAITS]
    handles.append(plt.Line2D([], [], marker="o", ls="", color=INK2, ms=5, markeredgecolor="white",
                              label="independent origin (stem branch)"))
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0), fontsize=6,
               handletextpad=0.3, columnspacing=1.2)
    fig.text(0.99, 0.995, "Silhouettes: game-icons.net (CC BY 3.0)", fontsize=5, color=MUTED, ha="right", va="top")
    save(fig, "Figure1")


# ---------------------------------------------------------------------------------------
# Figure 1: method overview
# ---------------------------------------------------------------------------------------

def figure1():           # saved as Figure2 (method overview)
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
    labels = ["Nucleotide rate", "Protein rate", "Protein − nucleotide rate", "Gene loss (TOGA)"]
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
    save(fig, "Figure2")


# ---------------------------------------------------------------------------------------
# Figure 2: rate layers on AliSim data
# ---------------------------------------------------------------------------------------

METHODS_RATE = [("naive", "Naive (RERconverge-style)", C1), ("permulation", "Control-lineage permutation", C2),
                ("naive_fisher", "Naive, Fisher over layers", C3), ("brownian_lmm", "Brownian phylo-LMM", C4),
                ("gls", "Learned-covariance GLS", C5), ("final", "GLS + leave-one-origin-out", C6)]


# kinds of convergent change, all for the final method (its colour), told apart by line and marker
KINDS = (("power_both", "Selection + rate", C6, "-", "o"), ("power_omega", "Selection (ω)", C6, "--", "s"),
         ("power_rate", "Rate only", C6, ":", "^"))


def main_benchmark():
    """Main rate benchmark: trees 0-2 (results/final_benchmark.csv) plus trees 3-9 (results/robustness)."""
    df = pd.read_csv(RES / "final_benchmark.csv")
    extra = [pd.read_csv(RES / "robustness" / f"main_tree{i}.csv") for i in range(3, 10)]
    return pd.concat([df] + [e[df.columns] for e in extra], ignore_index=True)


def origins_benchmark():
    return pd.concat([pd.read_csv(RES / "origins_power.csv"),
                      pd.read_csv(RES / "robustness" / "origins_power_extra.csv")], ignore_index=True)


def tree_spread(df, keys, col):
    """Mean over trees of the per-tree mean, with the range across trees."""
    per = df.groupby(keys + ["tree"])[col].mean()
    g = per.groupby(level=list(range(len(keys))))
    return g.mean(), g.min(), g.max()


def figure2():
    df = main_benchmark()
    og = origins_benchmark()
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.9), gridspec_kw=dict(hspace=0.6, wspace=0.42))
    conf = sorted(df.confounding.unique())
    x = np.arange(len(conf))
    w = 0.13

    ax = axes[0, 0]
    null, lo, hi = tree_spread(df[df.effect == "weak"], ["confounding", "method"], "false_calls_null")
    for i, (m, lab, c) in enumerate(METHODS_RATE):
        v, a, b = (np.array([s.get((cf, m), np.nan) for cf in conf]) for s in (null, lo, hi))
        ax.vlines(x + (i - 2.5) * w, a + 1, b + 1, color=c, lw=0.7, zorder=2)
        ax.scatter(x + (i - 2.5) * w, v + 1, color=c, s=14, zorder=3, label=lab, edgecolor="white", lw=0.5)
    ax.set_yscale("log")
    ax.set_xticks(x, [f"{c:g}" for c in conf])
    ax.set_xlabel("Rate-jump intensity")
    ax.set_ylabel("False calls + 1")
    grid(ax)
    panel(ax, "a", "No convergence")

    ax = axes[0, 1]
    fdr, lo, hi = tree_spread(df, ["confounding", "method"], "fdr")
    for i, (m, lab, c) in enumerate(METHODS_RATE):
        v, a, b = (np.array([s.get((cf, m), np.nan) for cf in conf]) for s in (fdr, lo, hi))
        ax.vlines(x + (i - 2.5) * w, a, b, color=c, lw=0.7, zorder=2)
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
        ax.plot(range(3), v, color=c, lw=1.1, ls=ls, marker=mk, ms=3, label=lab)
    ax.set_xticks(range(3), eff)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Power (final test)")
    ax.set_xlabel("Effect size")
    ax.legend(loc="lower right")
    grid(ax)
    panel(ax, "c", "Power by kind of change")

    ax = axes[1, 0]
    m, lo, hi = (v.reindex(eff) for v in tree_spread(fin, ["effect"], "signature_accuracy"))
    ax.errorbar(range(3), m, yerr=[m - lo, hi - m], color=C6, lw=1.1, marker="o", ms=3, capsize=3)
    ax.set_xticks(range(3), eff)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Exact signature accuracy")
    ax.set_xlabel("Effect size")
    grid(ax)
    panel(ax, "d", "Signatures of called genes")

    ax = axes[1, 1]
    fl = og[og.method == "gls_looo_acat"].groupby("origins")
    for col, lab, c, ls, mk in KINDS:
        ax.plot(fl[col].mean().index, fl[col].mean().values, color=c, lw=1.1, ls=ls, marker=mk, ms=3, label=lab)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Independent origins")
    ax.set_ylabel("Power (final test)")
    grid(ax)
    panel(ax, "e", "Power by number of origins")

    ax = axes[1, 2]
    for m, lab, c in (("naive:rate_nucleotide", "Naive", C1), ("gls_acat", "Learned-covariance GLS", C5),
                      ("gls_looo_acat", "GLS + leave-one-origin-out", C6)):
        s = og[og.method == m].groupby("origins").fdr.mean()
        ax.plot(s.index, s.values, color=c, lw=1.1, marker="o", ms=3, label=lab)
    ax.axhline(0.1, color=INK2, lw=0.8, ls="--")
    ax.set_ylim(-0.03, 1)
    ax.set_xlabel("Independent origins")
    ax.set_ylabel("Realised FDR (q = 0.1)")
    grid(ax)
    panel(ax, "f", "False discoveries by origins")

    handles = [plt.Line2D([], [], marker="o", ls="", color=c, ms=5, label=lab) for _, lab, c in METHODS_RATE]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    save(fig, "ExtendedDataFigure1")


SENS = [("Baseline", dict(tips=200, frac=0.02, discordance=0.0)),
        ("100 species", dict(tips=100)), ("400 species", dict(tips=400)),
        ("0.5% convergent", dict(frac=0.005)), ("5% convergent", dict(frac=0.05)),
        ("10% convergent", dict(frac=0.10)),
        ("Discordance 10", dict(discordance=10.0)), ("Discordance 40", dict(discordance=40.0))]
SENS_FILES = {"100 species": "tips100", "400 species": "tips400", "0.5% convergent": "frac0.005",
              "5% convergent": "frac0.05", "10% convergent": "frac0.10",
              "Discordance 10": "disc10", "Discordance 40": "disc40"}
SENS_METHODS = [("naive", "Naive", C1), ("gls", "Learned-covariance GLS", C5),
                ("final", "GLS + leave-one-origin-out", C6)]


def sensitivity_data():
    """Confounding 1: baseline from the 10 main trees, every other setting from 5 trees."""
    base = main_benchmark()
    out = [base[base.confounding == 1].assign(setting="Baseline")]
    for lab, key in SENS_FILES.items():
        d = pd.concat([pd.read_csv(f) for f in sorted((RES / "robustness").glob(f"sens_{key}_[0-9].csv"))])
        out.append(d.assign(setting=lab))
    return pd.concat(out, ignore_index=True)


def ed_sensitivity():
    d = sensitivity_data()
    labels = [lab for lab, _ in SENS]
    y = np.arange(len(labels))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True, gridspec_kw=dict(wspace=0.12))
    specs = [("false_calls_null", "False calls, no convergence", "a", "Null genomes", True),
             ("fdr", "Realised FDR (q = 0.1)", "b", "False discoveries", False),
             ("power", "Power (all effects)", "c", "Power", False)]
    for ax, (col, xl, letter, title, logx) in zip(axes, specs):
        sub = d if col != "false_calls_null" else d[d.effect == "weak"]
        mean, lo, hi = tree_spread(sub, ["setting", "method"], col)
        for i, (m, lab, c) in enumerate(SENS_METHODS):
            off = (1 - i) * 0.22
            v, a, b = (np.array([s.get((L, m), np.nan) for L in labels]) for s in (mean, lo, hi))
            sh = 1 if logx else 0
            ax.hlines(y + off, a + sh, b + sh, color=c, lw=0.7, zorder=2)
            ax.scatter(v + sh, y + off, color=c, s=12, zorder=3, edgecolor="white", lw=0.5, label=lab)
        if logx:
            ax.set_xscale("log")
            xl = "False calls + 1"
        elif col == "fdr":
            ax.axvline(0.1, color=INK2, lw=0.8, ls="--")
            ax.set_xlim(-0.03, 1)
        else:
            ax.set_xlim(0, 1.02)
        ax.set_xlabel(xl)
        for k in range(1, len(labels)):
            ax.axhline(k - 0.5, color=GRID, lw=0.5, zorder=0)
        grid(ax, axis="x")
        panel(ax, letter, title)
    axes[0].set_yticks(y, labels)
    axes[0].set_ylim(-0.6, len(labels) - 0.4)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, ms=5, label=lab) for _, lab, c in SENS_METHODS]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.55, -0.1))
    save(fig, "ExtendedDataFigure3")


# ---------------------------------------------------------------------------------------
# Figure 3: gene-loss layer
# ---------------------------------------------------------------------------------------

LOSS_METHODS = [("simulation", "This method (simulated null)", C1), ("gls", "Same statistic, GLS p-value", C2),
                ("fisher", "Fisher exact, no phylogeny", C3)]
# loss-null development on matched control trait sets of real genomes (README, "Gene-loss layer"
# and "Simulation null" sections; final row pooled over the five traits of the paper dataset)
LOSS_NULLS = [("GLS t-test", 60, 5.1), ("Control sets, 200 available", 20, 0.25),
              ("Control sets, 142 available", 10, 0.15), ("Simulated, species error", 5, 0.05),
              ("Simulated, species \u00d7 gene error", 13, 0.26)]


def figure3():
    d = pd.read_csv(RES / "loss_benchmark.csv")
    sig = d[d.penetrance > 0]
    origins = sorted(d.origins.unique())
    fig = plt.figure(figsize=(7.2, 6.2))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.9], hspace=0.65, wspace=0.18)
    letters = iter("abcdef")
    for row, (col, ylab, ylim) in enumerate((("power", "Power", (0, 1)), ("fdp", "False discovery proportion", (0, 0.75)))):
        for j, k in enumerate(origins):
            ax = fig.add_subplot(gs[row, j])
            for m, lab, c in LOSS_METHODS:
                g = sig[(sig.origins == k) & (sig.method == m)].groupby("penetrance")[col]
                ax.plot(g.mean().index, g.mean().values, color=c, lw=1.1, marker="o", ms=3,
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
    fig.legend(handles=[plt.Line2D([], [], color=c, lw=1.1, marker="o", ms=3, label=lab) for _, lab, c in LOSS_METHODS],
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
    ax.set_yticks(range(len(genes)), genes, fontsize=5)
    ax.set_xticks([])
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_xlabel(f"{len(order)} species in the order of the Zoonomia tree", fontsize=6.5)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    panel(ax, letter, title, pad=14)


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
    ax.set_yticks(range(len(genes)), genes, fontsize=5)
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
    fig = plt.figure(figsize=(183 * MM, 186 * MM))
    gs = fig.add_gridspec(3, 3, height_ratios=[3.6, 0.62, 1.15], width_ratios=[1, 1.25, 0.9], hspace=0.22,
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
    bottom_row = gs[2, :].subgridspec(1, 3, width_ratios=[1, 1, 1.2], wspace=0.35)
    ax = fig.add_subplot(bottom_row[0, 0])
    joint_panel(ax, mar, "beta_marine", "beta_semiaquatic", mg, ["PON1", "MMP12"], "d",
                "Marine vs semi-aquatic", "Partial effect: marine", "Partial effect: semi-aquatic")
    ax.legend(loc="center left", bbox_to_anchor=(1.15, 0.5), fontsize=6, markerscale=1.2)
    save(fig, "ExtendedDataFigure8")


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
    save(fig, "ExtendedDataFigure10")


# ---------------------------------------------------------------------------------------
# Supplementary Figure 1: RERconverge
# ---------------------------------------------------------------------------------------

def supp1():
    r = pd.read_csv(RES / "rerconverge_compare.csv")
    # 10,000 permulations on all genes, rate-jump intensity 1 only (results/robustness)
    allp = pd.concat([pd.read_csv(RES / "robustness" / f) for f in ("rerconverge_allperm10k.csv",
                                                                     "rerconverge_allperm10k_conf02.csv")])
    allp = allp[allp.method == "RERconverge_protein_permulation"].assign(method="RERconverge_allperm")
    r = pd.concat([r, allp], ignore_index=True)
    meth = [("final", "This method (final)", C6), ("RERconverge_protein", "RERconverge, protein", C1),
            ("RERconverge_nucleotide", "RERconverge, nucleotide", C2),
            ("RERconverge_protein_permulation", "+ permulations, 150 candidates", "#6250d6"),   # violet, slot 7
            ("RERconverge_allperm", "+ permulations, all genes", "#b3a8ef")]
    conf = sorted(r.confounding.unique())
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw=dict(wspace=0.45))
    x = np.arange(len(conf))
    w = 0.16
    for ax, (eff, col, ylab, letter, title) in zip(axes, (
            ("none", "false_calls", "False calls", "a", "No convergence"),
            ("moderate", "fdr", "Realised FDR (q = 0.1)", "b", "2% of genes convergent"),
            ("moderate", "power", "Power", "c", "2% of genes convergent"))):
        s = r[r.effect == eff].groupby(["confounding", "method"])[col].mean()
        for i, (m, lab, c) in enumerate(meth):
            v = [s.get((cf, m), np.nan) for cf in conf]
            ax.bar(x + (i - 2) * w, v, width=w - 0.02, color=c, label=lab)
            for xx, vv in zip(x + (i - 2) * w, v):
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
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.3))
    save(fig, "ExtendedDataFigure2")


# ---------------------------------------------------------------------------------------
# Figure 7: marine losses in detail (lineage funnels, PON1 mutation map, exon synteny)
# ---------------------------------------------------------------------------------------

MUT_STYLE = {"STOP": ("Stop codon", C2, "v"), "FS": ("Frameshift", C5, "D"),
             "SSM": ("Splice-site mutation", C3, "s")}
DETAIL_GENE = "PON1"
ORIGIN_SETS = [("Cetaceans", "Cetacea", ["Hippopotamus_amphibius", "Bos_taurus", "Sus_scrofa", "Camelus_dromedarius"]),
               ("Pinnipeds", "Pinniped", ["Pteronura_brasiliensis", "Mellivora_capensis", "Canis_lupus_familiaris", "Felis_catus"]),
               ("Sirenians", "Sirenia", ["Loxodonta_africana", "Procavia_capensis", "Heterohyrax_brucei", "Elephantulus_edwardii"])]
DETAIL_NAMES = {"Eubalaena_japonica": "right whale", "Balaenoptera_bonaerensis": "minke whale",
                "Mesoplodon_bidens": "Sowerby's beaked whale", "Lipotes_vexillifer": "baiji",
                "Tursiops_truncatus": "bottlenose dolphin", "Mirounga_angustirostris": "N. elephant seal",
                "Trichechus_manatus": "W. Indian manatee", "Loxodonta_africana": "African elephant",
                "Heterohyrax_brucei": "yellow-spotted hyrax", "Elephantulus_edwardii": "Cape elephant shrew",
                "Zalophus_californianus": "California sea lion", "Neomonachus_schauinslandi": "Hawaiian monk seal"}
SYNTENY_PAIRS = [("Bos_taurus", "Tursiops_truncatus"), ("Canis_lupus_familiaris", "Zalophus_californianus"),
                 ("Loxodonta_africana", "Trichechus_manatus")]


def mut_kind(t):
    return "FS" if t.startswith("FS_") else t


def detail_data(gene):
    m = pd.read_csv(RES / "extra" / "marine_detail_mutations.csv")
    e = pd.read_csv(RES / "extra" / "marine_detail_exons.csv")
    e = e[e.gene == gene]
    chain = e.groupby("species").chain.first().astype(str)
    m = m[(m.gene == gene) & (m.type != "COMPENSATION")]
    m = m[m.chain.astype(str) == m.species.map(chain)]
    # reference exon lengths (bp): the most common projected structure among the terrestrial relatives
    rel = e[e.group == "relative"]
    struct = []
    for _, g in rel.groupby("species"):
        L = (g.end - g.start).tolist()
        struct.append(tuple(L if g.strand.iloc[0] == "+" else L[::-1]))
    ref = np.array(pd.Series(struct).mode()[0])
    return m, e, ref


def query_exons(e, m, sp, n_exons):
    """Exon number -> (start, end) in kb from the locus start, in transcription direction."""
    g = e[e.species == sp].sort_values("start")
    st, en = g.start.to_numpy(), g.end.to_numpy()
    if g.strand.iloc[0] == "-":
        st, en = (en.max() - en)[::-1], (en.max() - st)[::-1]
    else:
        st, en = st - st.min(), en - st.min()
    gone = set(m[(m.species == sp) & m.type.isin(["Deleted exon", "Missing exon"])].exon)
    nums = [k for k in range(1, n_exons + 1) if k not in gone][: len(st)]
    return {k: (a / 1e3, b / 1e3) for k, a, b in zip(nums, st, en)}


def cladogram(tree):
    """Copy of the tree with tips aligned: node depth = levels above the deepest tip."""
    h = np.zeros(tree.n_nodes)
    for v in range(tree.n_nodes - 1):
        h[tree.parent[v]] = max(h[tree.parent[v]], h[v] + 1)
    length = np.where(tree.parent >= 0, h[np.maximum(tree.parent, 0)] - h, 0.0)
    return type(tree)(parent=tree.parent, length=length.astype(float), n_tips=tree.n_tips)


def labelled_tree(ax, tree, names, common, marine, fs=5.6):
    t = cladogram(tree)
    x, y = draw_tree(ax, t, {v: C1 for v in np.nonzero(foreground_branches(
        t, np.array([nm in marine for nm in names])))[0]}, lw=1.1)
    for i in range(t.n_tips):
        ax.text(x[i] + 0.25, y[i], common.get(names[i], names[i]), va="center", fontsize=fs,
                color=INK if names[i] in marine else INK2)
    return x, y


def funnel_panel(axt, axf, row, letter, title, tree, names, common, marine):
    x, _ = labelled_tree(axt, tree, names, common, marine, fs=5.3)
    axt.set_xlim(-0.2, x.max() + 0.75 * x.max() + 6.5)
    axt.set_ylim(tree.n_tips - 0.5, -0.5)
    panel(axt, letter, title)
    steps = [("genes analysed", row.genes_analysed), ("lost in ≥1 species", row.lost_any),
             ("lost in all species", row.lost_all), ("…intact in sister", row.origin_specific)]
    axf.set_axis_off()
    axf.set_xlim(0, 1)
    axf.set_ylim(5.2, -0.3)
    for i, (lab, n) in enumerate(steps):
        axf.add_patch(FancyBboxPatch((0.04, i + 0.08), 0.92, 0.62, boxstyle="round,pad=0,rounding_size=0.06",
                                     fc=PALE, ec=GRID, lw=0.6, mutation_aspect=4))
        axf.text(0.5, i + 0.3, f"{n:,}", ha="center", va="center", fontsize=6.8, color=INK, fontweight="bold")
        axf.text(0.5, i + 0.56, lab, ha="center", va="center", fontsize=5, color=INK2)
        if i:
            axf.annotate("", (0.5, i + 0.08), (0.5, i - 0.3), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.6))
    axf.add_patch(FancyBboxPatch((0.04, 4.28), 0.92, 0.82, boxstyle="round,pad=0,rounding_size=0.06",
                                 fc=C1, ec=C1, lw=0.6, mutation_aspect=4))
    axf.text(0.5, 4.52, f"{row.convergent_calls}", ha="center", va="center", fontsize=6.8, color="white",
             fontweight="bold")
    axf.text(0.5, 4.85, "convergent", ha="center", va="center", fontsize=5, color="white")
    axf.annotate("", (0.5, 4.28), (0.5, 3.7), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.6, ls=":"))


def marine_context():
    import convergence_toga as CT
    from mlconv.tree import subtree
    full, names_all = from_newick((RES / "toga" / "master_paper.nwk").read_text())
    common = pd.read_csv(RES / "toga" / "assemblies_paper.tsv", sep="\t").set_index("species")["Common name"].to_dict()
    common.update(SHORT_NAMES)
    common.update(DETAIL_NAMES)
    cov = pd.read_csv(RES / "toga" / "coverage_paper.tsv", sep="\t").drop_duplicates("species").set_index("species")
    lineage = cov.lineage.reindex(names_all).fillna("")
    marine = {s for s in names_all if CT.TRAITS["marine"][0](s, lineage[s])}
    m, e, ref = detail_data(DETAIL_GENE)
    origin_of, sets = {}, []
    for label, key, rel in ORIGIN_SETS:
        ms = [s for s in names_all if s in marine and (key in lineage[s] or (key == "Pinniped" and any(
            k in lineage[s] for k in ("Otariidae", "Phocidae", "Odobenidae"))))]
        origin_of.update({s: label for s in ms})
        sets.append((label, ms, rel))
    return dict(full=full, names=names_all, common=common, marine=marine, m=m, e=e, ref=ref,
                bounds=np.concatenate([[0], np.cumsum(ref)]) / 3, origin_of=origin_of, sets=sets, subtree=subtree)


def funnels_into(fig, cell, ctx, letters="abc"):
    fun = pd.read_csv(RES / "extra" / "marine_origin_funnel.csv").set_index("origin")
    top = cell.subgridspec(1, 6, width_ratios=[1.5, 0.66] * 3, wspace=0.02)
    for i, (label, ms, rel) in enumerate(ctx["sets"]):
        t, nm = ctx["subtree"](ctx["full"], ctx["names"], ms + rel)
        funnel_panel(fig.add_subplot(top[2 * i]), fig.add_subplot(top[2 * i + 1]), fun.loc[label], letters[i], label,
                     t, nm, ctx["common"], ctx["marine"])


def mutation_map_into(fig, cell, ctx, letter, fs=5.2):
    m, bounds, origin_of, common, marine = ctx["m"], ctx["bounds"], ctx["origin_of"], ctx["common"], ctx["marine"]
    n_ex = len(ctx["ref"])
    mid = cell.subgridspec(1, 2, width_ratios=[1.35, 3], wspace=0.1)
    keep = [s for s in ctx["names"] if s in origin_of] + [s for _, _, r in ctx["sets"] for s in r]
    t, nm = ctx["subtree"](ctx["full"], ctx["names"], keep)
    at = fig.add_subplot(mid[0])
    tx, ty = labelled_tree(at, t, nm, common, marine, fs=fs)
    at.set_xlim(-0.2, tx.max() + 10.5)
    at.set_ylim(t.n_tips - 0.5, -2.2)
    at.text(tx.max() + 10.3, -1.3, f"Human {DETAIL_GENE} exons", ha="right", va="center", fontsize=fs, color=INK)
    ax = fig.add_subplot(mid[1], sharey=at)
    ax.patch.set_alpha(0)
    row = {s: ty[i] for i, s in enumerate(nm)}
    for k in range(n_ex):
        ax.add_patch(Rectangle((bounds[k], -1.75), bounds[k + 1] - bounds[k], 0.9, fc=C1 if k % 2 == 0 else "#9cc3ef",
                               ec="white", lw=0.8))
        ax.text((bounds[k] + bounds[k + 1]) / 2, -1.3, str(k + 1), ha="center", va="center", fontsize=5,
                color="white" if k % 2 == 0 else INK)
    for s, yy in row.items():
        ax.plot([0, bounds[-1]], [yy, yy], color=GRID, lw=0.6, zorder=1)
    ms = m[m.species.isin(row)]
    for r in ms.itertuples():
        yy = row[r.species]
        if r.type in ("Deleted exon", "Missing exon"):
            a, b = bounds[r.exon - 1], bounds[r.exon]
            ax.add_patch(Rectangle((a, yy - 0.32), b - a, 0.64, fc=C4 if r.type == "Deleted exon" else "white",
                                   ec=C4 if r.type == "Deleted exon" else MUTED,
                                   hatch=None if r.type == "Deleted exon" else "////", lw=0.6, zorder=2))
            continue
        k = mut_kind(r.type)
        if k not in MUT_STYLE:
            continue
        _, col, mk = MUT_STYLE[k]
        xx = bounds[r.exon - 1] if r.codon == 0 else r.codon - 0.5
        ax.scatter(xx, yy, s=10, marker=mk, color="white" if r.masked else col, edgecolor=col, lw=0.7, zorder=3)
    ms = ms.assign(origin=ms.species.map(origin_of), kind=ms.type.map(mut_kind)).dropna(subset=["origin"])
    ms = ms[ms.kind.isin(MUT_STYLE)]
    for (o, k, ex), g in ms.groupby(["origin", "kind", "exon"]):
        for c in sorted(set(g.codon)):
            h = g[(g.codon - c).abs() <= 1]
            if h.species.nunique() >= 2 and c == h.codon.min():
                ys_ = [row[s] for s in h.species]
                xx = bounds[ex - 1] if c == 0 else c - 0.5
                ax.plot([xx, xx], [min(ys_), max(ys_)], color=MUTED, lw=0.6, zorder=2.5)
    ax.set_xlim(-2, bounds[-1] + 2)
    ax.set_xlabel("Reference codon")
    ax.spines[["left"]].set_visible(False)
    ax.tick_params(left=False, labelleft=False)
    handles = [plt.Line2D([], [], marker=mk, ls="", color=c, ms=3.5, label=lab) for lab, c, mk in MUT_STYLE.values()]
    handles += [Rectangle((0, 0), 1, 1, fc=C4, ec=C4, label="Deleted exon"),
                Rectangle((0, 0), 1, 1, fc="white", ec=MUTED, hatch="////", label="Missing exon (gap)"),
                plt.Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, ms=3.5, label="Masked by TOGA"),
                plt.Line2D([], [], color=MUTED, lw=0.6, label="Shared within origin")]
    ax.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(0.4, -0.1), fontsize=5,
              handlelength=1.2, columnspacing=1.0)
    panel(at, letter, f"Inactivating mutations in {DETAIL_GENE}")
    return at, ax


def synteny_into(fig, cell, ctx, letter):
    m, e, bounds, common, marine = ctx["m"], ctx["e"], ctx["bounds"], ctx["common"], ctx["marine"]
    n_ex = len(ctx["ref"])
    bot = cell.subgridspec(1, 3, wspace=0.15)
    for i, (a_sp, b_sp) in enumerate(SYNTENY_PAIRS):
        ax = fig.add_subplot(bot[i])
        A, B = query_exons(e, m, a_sp, n_ex), query_exons(e, m, b_sp, n_ex)
        span = max(max(v[1] for v in A.values()), max(v[1] for v in B.values()))
        for yy, ex, sp in ((1, A, a_sp), (0, B, b_sp)):
            ax.plot([0, max(v[1] for v in ex.values())], [yy, yy], color=MUTED, lw=0.7, zorder=1)
            for k, (s0, s1) in ex.items():
                ax.add_patch(Rectangle((s0, yy - 0.07), max(s1 - s0, span * 0.012), 0.14,
                                       fc=C1 if sp in marine else MUTED, ec="none", zorder=2))
            ax.text(0, yy + (0.19 if yy else -0.42), f"{common.get(sp, sp)} ({max(v[1] for v in ex.values()):.1f} kb)",
                    fontsize=5.5, va="center", color=INK if sp in marine else INK2)
        for k in A:
            if k in B:
                (a0, a1), (b0, b1) = A[k], B[k]
                ax.fill([a0, a1, b1, b0], [0.93, 0.93, 0.07, 0.07], color=C4, alpha=0.45, lw=0, zorder=0)
        for r in m[m.species == b_sp].itertuples():
            k = mut_kind(r.type)
            if k not in MUT_STYLE or r.exon not in B:
                continue
            b0, b1 = B[r.exon]
            frac = 0 if r.codon == 0 else (r.codon - bounds[r.exon - 1]) / (bounds[r.exon] - bounds[r.exon - 1])
            _, col, mk = MUT_STYLE[k]
            ax.scatter(b0 + np.clip(frac, 0, 1) * (b1 - b0), -0.17, s=10, marker=mk,
                       color="white" if r.masked else col, edgecolor=col, lw=0.7, zorder=3)
        ax.set_xlim(-span * 0.02, span * 1.02)
        ax.set_ylim(-0.55, 1.35)
        ax.set_yticks([])
        ax.spines[["left"]].set_visible(False)
        ax.set_xlabel("kb from first coding exon")
        if i == 0:
            panel(ax, letter, f"{DETAIL_GENE} exons in terrestrial and marine genomes")


# ---------------------------------------------------------------------------------------
# Figure 8: alignments of convergent losses, genome-wide volcano plots, functional map
# ---------------------------------------------------------------------------------------

FUNCTION_GROUPS = {
    "marine": [("Smell", ["OR7A17", "OR5M9", "OR2T1", "OR51G1", "OR5H15", "OR9Q2", "OR5C1", "OR5M3", "OR6C2", "OR51I1",
                          "OR51D1", "OR51I2", "OR5H1", "OR51A7", "OR2K2", "OR5H14", "TAAR5"]),
               ("Taste", ["GNAT3", "CALHM1", "TAS1R3", "TAS2R1"]),
               ("Skin and hair", ["KRT1", "KRT10", "KRT79", "KRTAP24-1", "KLK8", "GSDMA", "SERPINB12", "ANXA9"]),
               ("Muscle", ["MYH6", "MYH13", "MYBPHL", "MSS51", "ANKRD23"]),
               ("Detoxification and metabolism", ["PON1", "ACSM4", "GLYATL3", "SULT6B1", "PTGR1", "GBA3"]),
               ("Other", ["SLC4A9", "ATP13A5", "TGM7", "HTRA4", "MRGPRX2", "ADGRF2", "CCDC170", "TLE7"])],
    "hypoxia_subterranean": [("Lens", ["LGSN", "LCTL", "CRYGS", "CRYBA4", "BFSP1", "GJA8"]),
                             ("Retina", ["MPP4", "ROM1", "RGS11", "CABP5"]),
                             ("Vitreous", ["OPTC"]),
                             ("Other", ["RSKR", "LACTBL1", "B3GNT4", "TPD52L3"])]}
REPORTED_BEFORE = {"PON1", "SLC4A9", "TAS1R3", "TAS2R1"}     # cited in the Results [11,30,31]
VOLCANO_TRAITS = [("marine", "Marine", C1), ("hypoxia_subterranean", "Subterranean", C3),
                  ("echolocation", "Echolocation", C2), ("hypoxia_altitude", "High altitude", C4)]


def loss_setup():
    import convergence_toga as CT
    tree, names = from_newick((RES / "toga" / "master_paper.nwk").read_text())
    cov = pd.read_csv(RES / "toga" / "coverage_paper.tsv", sep="\t").drop_duplicates("species").set_index("species")
    lineage = cov.lineage.reindex(names).fillna("")
    m = pd.read_csv(RES / "toga" / "loss_status_paper.tsv.gz", sep="\t", index_col=0).reindex(columns=names).fillna("M")
    Y = pd.DataFrame(np.where(m.isin(["L", "UL"]), 1.0, np.where(m.isin(["I", "PI"]), 0.0, np.nan)),
                     index=m.index, columns=names)
    trait = {k: np.array([CT.TRAITS[k][0](s, lineage[s]) for s in names]) for k, _, _ in VOLCANO_TRAITS}
    return tree, names, Y, trait


def volcano_panel(ax, key, label, colour, Y, trait, letter, n_label=8):
    d = pd.read_csv(RES / f"{key}_toga" / f"{key}_genes.csv").dropna(subset=["p_loss"]).set_index("symbol")
    d = d[~d.index.duplicated()]
    t = trait[key]
    Yg = Y.reindex(d.index)
    diff = np.nanmean(Yg.loc[:, t], 1) - np.nanmean(Yg.loc[:, ~t], 1)
    y = -np.log10(d.p_loss.clip(lower=1e-12))
    call = d.call_loss.astype(bool).to_numpy()
    ax.scatter(diff[~call], y[~call], s=3, color=MUTED, alpha=0.35, lw=0, rasterized=True)
    ax.scatter(diff[call], y[call], s=9, color=colour, lw=0, zorder=3)
    if call.any():
        thr = y[call].min()
        ax.axhline(thr, color=INK2, lw=0.6, ls="--")
    ymax = max(5.0, float(y.max()) + 0.4)
    top = d[call].sort_values("p_loss").index[:n_label]
    for i, g in enumerate(top):
        j = d.index.get_loc(g)
        ax.annotate(g, (diff[j], y.iloc[j]), xytext=(-0.52, ymax * (0.86 - 0.075 * i)), textcoords="data",
                    fontsize=5, color=INK, ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.35, shrinkA=1, shrinkB=1))
    ax.set_xlim(-0.6, 1.05)
    ax.set_ylim(0, ymax * 1.08)
    ax.set_xlabel("Loss in trait − other species", fontsize=6.5)
    ax.tick_params(labelsize=6)
    grid(ax)
    n = int(call.sum())
    panel(ax, letter, f"{label} ({n})")


def origin_losses(tree, names, Y, t, genes):
    """gene -> list of booleans, one per origin: lost in at least one species of that origin."""
    from mlconv.tree import origin_nodes
    D = tree.tip_sets()
    out = {}
    for g in genes:
        y = Y.loc[g].to_numpy() if g in Y.index else np.full(len(names), np.nan)
        out[g] = [bool(np.nansum(y[D[o]] == 1) > 0) for o in origin_nodes(tree, t)]
    return out


def function_map(ax, tree, names, Y, trait, letter):
    ax.set_axis_off()
    ax.set_xlim(0, 100)
    H = 60
    ax.set_ylim(0, H)
    panel(ax, letter, "Functions of the convergent losses")
    blocks = [("marine", "Marine (48 genes)", C1, ["cetaceans", "pinnipeds", "manatee"], 0, 61),
              ("hypoxia_subterranean", "Subterranean (15 genes)", C3, None, 63, 100)]
    for key, title, colour, onames, x0, x1 in blocks:
        groups = FUNCTION_GROUPS[key]
        genes = [g for _, gs in groups for g in gs]
        ol = origin_losses(tree, names, Y, trait[key], genes)
        n_or = len(next(iter(ol.values())))
        ax.add_patch(FancyBboxPatch((x0 + 0.3, 0.3), x1 - x0 - 0.6, H - 1.4, boxstyle="round,pad=0,rounding_size=1.2",
                                    fc="none", ec=colour, lw=1.0))
        ax.text(x0 + 1.5, H - 2.7, title, fontsize=7.5, color=colour, fontweight="bold", va="center")
        cols = 5 if key == "marine" else 2
        cw = (x1 - x0 - 3) / cols
        y = H - 6.0
        for name, gs in groups:
            ax.text(x0 + 1.5, y, name, fontsize=6.3, color=INK, va="center", fontweight="bold")
            y -= 2.6
            for i, g in enumerate(gs):
                cx = x0 + 1.5 + (i % cols) * cw
                cy = y - (i // cols) * 2.9
                ax.add_patch(FancyBboxPatch((cx, cy - 1.0), cw - 0.6, 2.2, boxstyle="round,pad=0,rounding_size=0.5",
                                            fc=PALE, ec=INK if g in REPORTED_BEFORE else GRID,
                                            lw=0.9 if g in REPORTED_BEFORE else 0.5))
                ax.text(cx + 0.6, cy + 0.1, g, fontsize=5.2, color=INK, va="center")
                for k, lost in enumerate(ol[g]):
                    ax.add_patch(plt.Circle((cx + cw - 1.4 - (n_or - 1 - k) * 0.75, cy + 0.1), 0.28,
                                            color=colour if lost else GRID, lw=0))
            y -= 2.9 * ((len(gs) - 1) // cols + 1) + 0.6
        note = (f"dots: lost in {', '.join(onames)}" if onames else f"dots: lost in each of the {n_or} origins")
        if onames:
            ax.text(x1 - 1.2, H - 2.7, note, fontsize=5.4, color=INK2, ha="right", va="center")
        else:
            ax.text(x0 + 1.5, 6.2, note, fontsize=5.4, color=INK2, va="center")
    ax.text(64.5, 1.6, "Black outline: loss reported\nin earlier studies", fontsize=5.4, color=INK2)


AL_STYLE = {"fs": C5, "stop": C2, "gap": PALE}


def alignment_panel(ax, gene, windows, species, common, marine_like, colour, letter, title, shared=(), fs=5):
    """Codon alignment: one label per codon (monospace); '.' marks a base identical to human."""
    a = pd.read_csv(RES / "extra" / "codon_alignment_detail.csv")
    a = a[a.gene == gene]
    ref = a.drop_duplicates("ref_pos").set_index("ref_pos").ref
    cols = []
    for w0, w1 in windows:
        cols += list(range(w0, w1 + 1)) + [None]
    cols = cols[:-1]
    ax.set_xlim(-0.6, len(cols) - 0.4)
    ax.set_ylim(len(species) + 0.5, -1.9)
    ax.set_axis_off()
    mono = dict(family="Liberation Mono", fontsize=fs, ha="center", va="center")
    by = {s_: g.set_index("ref_pos")["query"] for s_, g in a.groupby("species")}
    for j, c in enumerate(cols):
        if c is None:
            ax.text(j, -0.8, "//", ha="center", va="center", fontsize=fs, color=MUTED)
            continue
        if c in shared:
            ax.add_patch(Rectangle((j - 0.5, 0.45), 1.0, len(species) + 0.1, fc="none", ec=colour, lw=0.8, ls="--"))
        if c % 5 == 0:
            ax.text(j, -1.55, str(c), ha="center", va="center", fontsize=fs, color=INK2)
        ax.text(j, -0.55, ref.get(c, "   "), color=INK, **mono)
    ax.text(-0.7, -0.55, "human", ha="right", va="center", fontsize=fs, color=INK)
    for i, s_ in enumerate(species):
        yy = i + 1
        ax.text(-0.7, yy, common.get(s_, s_), ha="right", va="center", fontsize=fs,
                color=colour if s_ in marine_like else INK2)
        q = by.get(s_)
        for j, c in enumerate(cols):
            if c is None:
                continue
            cod = q.get(c, "---") if q is not None else "   "
            r_ = ref.get(c, "   ")
            if cod == "XXX":
                ax.add_patch(Rectangle((j - 0.42, yy - 0.42), 0.84, 0.84, fc=AL_STYLE["fs"], ec="none"))
                ax.text(j, yy, "fs", color="white", fontweight="bold", **mono)
                continue
            stop = cod in ("TAA", "TAG", "TGA") and c < ref.index.max()
            if stop:
                ax.add_patch(Rectangle((j - 0.42, yy - 0.42), 0.84, 0.84, fc=AL_STYLE["stop"], ec="none"))
            shown = "".join("." if (k < len(cod) and cod[k] == r_[k]) else (cod[k] if k < len(cod) else " ")
                            for k in range(3))
            ax.text(j, yy, shown, color="white" if stop else (INK2 if shown == "..." else INK), **mono)
    panel(ax, letter, title)


# ---------------------------------------------------------------------------------------
# Supplementary Figure 3: how many origins the loss test needs (subterranean trait)
# ---------------------------------------------------------------------------------------

SIGHTED = {"Ctenomys", "Ellobius"}          # subterranean lineages that keep functional eyes


def supp3():
    d = pd.read_csv(RES / "extra" / "origin_subsampling.csv")
    parts = d.subset.str.split(";")
    d["sighted"] = parts.map(lambda p: sum(x in SIGHTED for x in p))
    d["blind"] = parts.map(lambda p: sum(x not in SIGHTED for x in p))
    d["new"] = d.calls - d.recovered
    cols = {0: C3, 1: C1, 2: C2}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw=dict(wspace=0.32))
    rng = np.random.default_rng(1)
    for ax, col, ylab, letter, title in ((axes[0], "recovered", "Calls of the full analysis recovered (of 15)", "a",
                                          "Recovery depends on blind lineages"),
                                         (axes[1], "new", "Calls not made with all six origins", "b", "Calls not in the full analysis")):
        for k, c in cols.items():
            g = d[d.sighted == k]
            x = g.blind + (k - 1) * 0.18 + rng.uniform(-0.05, 0.05, len(g))
            ax.scatter(x, g[col], s=22, color=c, edgecolor="white", lw=0.7, zorder=3,
                       label=f"{k} sighted lineage{'s' if k != 1 else ''} kept")
        ax.set_xticks(range(1, 5))
        ax.set_xlabel("Blind subterranean lineages kept (of 4)")
        ax.set_ylabel(ylab, fontsize=7)
        grid(ax)
        panel(ax, letter, title)
    axes[0].axhline(15, color=INK2, lw=0.6, ls="--")
    axes[0].legend(loc="center left", bbox_to_anchor=(0.0, 0.62), fontsize=6)
    save(fig, "ExtendedDataFigure9")


# ---------------------------------------------------------------------------------------
# Figure 9: tissue, pathway and GO enrichment of the convergent losses
# ---------------------------------------------------------------------------------------

def enrichment_bars(ax, df, colour, letter, title, n=7, max_size=700, rename=None):
    d = df[df["size"] <= max_size].sort_values("size").drop_duplicates("genes")    # most specific term per gene set
    d = d.sort_values(["q", "p"]).head(n).iloc[::-1]
    y = np.arange(len(d))
    sig = d.q < 0.1
    ax.barh(y, d.called_in_term, color=[colour if v else GRID for v in sig], height=0.62)
    ax.scatter(d.expected_matched, y, marker="|", s=90, color=INK, lw=1.4, zorder=3)
    for yy, (_, r) in zip(y, d.iterrows()):
        ax.text(r.called_in_term + 0.25, yy, f"q = {r.q:.1g}" if r.q < 0.1 else f"p = {r.p:.1g}",
                va="center", fontsize=5.6, color=INK if r.q < 0.1 else INK2)
    names_ = [(rename or {}).get(n_, n_) for n_ in d.name]
    labels = [(n_[0].upper() + n_[1:]) if len(n_) <= 38 else (n_[0].upper() + n_[1:36] + "\u2026") for n_ in names_]
    ax.set_yticks(y, labels, fontsize=6)
    ax.set_xlim(0, d.called_in_term.max() * 1.35 + 1)
    ax.set_xlabel("Called genes in term (tick: expected)", fontsize=6.5)
    ax.tick_params(axis="x", labelsize=6)
    grid(ax, "x")
    panel(ax, letter, title)


def figure9():
    tis = pd.read_csv(RES / "extra" / "enrichment_tissue.csv")
    rea = pd.read_csv(RES / "extra" / "enrichment_reactome.csv")
    go = pd.read_csv(RES / "extra" / "enrichment_go.csv")
    fig, axes = plt.subplots(3, 2, figsize=(174 * MM, 172 * MM), gridspec_kw=dict(wspace=0.95, hspace=0.62))
    for j, (k, lab, c) in enumerate((("marine", "Marine", C1), ("hypoxia_subterranean", "Subterranean", C3))):
        enrichment_bars(axes[0, j], tis[tis.trait == k], c, "ab"[j], f"{lab}: tissue (Human Protein Atlas)", n=5)
        r = rea[rea.trait == k]
        if len(r):
            enrichment_bars(axes[1, j], r, c, "cd"[j], f"{lab}: Reactome pathways", n=5)
        else:
            axes[1, j].set_axis_off()
            axes[1, j].text(0.5, 0.5, "No pathway with ≥2 called genes", ha="center", va="center", fontsize=7,
                            color=INK2, transform=axes[1, j].transAxes)
            panel(axes[1, j], "cd"[j], f"{lab}: Reactome pathways")
        enrichment_bars(axes[2, j], go[go.trait == k], c, "ef"[j], f"{lab}: GO biological process", n=6, max_size=450)
    save(fig, "ExtendedDataFigure5")


# ---------------------------------------------------------------------------------------
# Figure 10: pathway maps of the senses lost (taste and smell in marine, the eye underground)
# ---------------------------------------------------------------------------------------

def gene_status(key, genes, tree, names, Y, trait):
    from mlconv.tree import origin_nodes
    d = pd.read_csv(RES / f"{key}_toga" / f"{key}_genes.csv").drop_duplicates("symbol").set_index("symbol")
    t = trait[key]
    D = tree.tip_sets()
    orig = origin_nodes(tree, t)
    out = {}
    for g in genes:
        y = Y.loc[g].to_numpy() if g in Y.index else np.full(len(names), np.nan)
        out[g] = dict(called=bool(d.call_loss.get(g, False)) if g in d.index else False,
                      lost=int(np.nansum(y[t] == 1)), n=int((~np.isnan(y[t])).sum()),
                      origins=[bool(np.nansum(y[D[o]] == 1) > 0) for o in orig])
    return out


def gene_node(ax, x, y, g, st, colour, w=9.0, h=3.2, fs=5.6):
    s = st[g]
    if s["called"]:
        fc, ec, tc, lw = colour, colour, "white", 0.8
    elif s["lost"] > 0:
        fc, ec, tc, lw = "white", colour, INK, 1.1
    else:
        fc, ec, tc, lw = PALE, GRID, INK2, 0.6
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.8",
                                fc=fc, ec=ec, lw=lw, zorder=3))
    ax.text(x, y + 0.35, g, ha="center", va="center", fontsize=fs, color=tc, zorder=4,
            fontweight="bold" if s["called"] else "normal")
    k = len(s["origins"])
    for i, lost in enumerate(s["origins"]):
        ax.add_patch(plt.Circle((x + (i - (k - 1) / 2) * 0.95, y - 0.95), 0.32, zorder=4,
                                color=(("white" if s["called"] else colour) if lost else (GRID if not s["called"] else "#ffffff55")),
                                lw=0))


def arrow(ax, a, b, style="-|>", ls="-", col=INK2, rad=0.0):
    ax.annotate("", b, a, arrowprops=dict(arrowstyle=style, color=col, lw=0.8, ls=ls, shrinkA=0, shrinkB=0,
                                          connectionstyle=f"arc3,rad={rad}"), zorder=2)


def note(ax, x, y, txt, **kw):
    ax.text(x, y, txt, fontsize=kw.pop("fs", 5.4), color=kw.pop("color", INK2), ha=kw.pop("ha", "center"),
            va=kw.pop("va", "center"), style=kw.pop("style", "italic"), zorder=4, **kw)


MAP_MARINE_GENES = ["TAS1R1", "TAS1R2", "TAS1R3", "TAS2R1", "TAS2R38", "GNAT3", "PLCB2", "ITPR3", "TRPM5", "CALHM1",
                    "CALHM3", "TAAR5", "OR7A17", "GNAL", "ADCY3", "CNGA2", "CNGA4", "CNGB1", "ANO2", "PON1", "PON2", "PON3"]
MAP_EYE_GENES = ["CRYAA", "CRYBA1", "CRYBA4", "CRYBB2", "CRYGC", "CRYGS", "BFSP1", "BFSP2", "LGSN", "MIP", "GJA3", "GJA8",
                 "LCTL", "OPTC", "RHO", "GNAT1", "PDE6A", "PDE6B", "CNGA1", "PRPH2", "ROM1", "MPP4", "GRM6", "GNAO1", "TRPM1",
                 "RGS11", "RGS7", "GNB5", "NYX", "CABP5"]


def map_axes(ax, ymax=62):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, ymax)
    ax.set_aspect("equal")
    ax.set_axis_off()


def map_legend(ax, c, y, dots_note):
    for i, (lab, fc, ec) in enumerate((("convergent loss (q = 0.1)", c, c), ("lost in some trait species", "white", c),
                                       ("intact in all trait species", PALE, GRID))):
        lx = 3 + i * 25
        ax.add_patch(FancyBboxPatch((lx, y - 1.2), 4.5, 2.4, boxstyle="round,pad=0,rounding_size=0.6", fc=fc, ec=ec,
                                    lw=0.9))
        ax.text(lx + 5.5, y, lab, fontsize=5.5, color=INK, va="center")
    ax.text(97, y - 3.2, dots_note, fontsize=5.2, color=INK2, va="center", ha="right")


def taste_map(ax, sm, letter, title="Marine mammals: taste, smell and detoxification"):
    c = C1
    map_axes(ax)
    panel(ax, letter, title)
    ax.add_patch(FancyBboxPatch((3, 4), 44, 50, boxstyle="round,pad=0,rounding_size=8", fc="#eef4fc", ec="#c9dcf5", lw=1))
    note(ax, 25, 51.5, "Taste receptor cell (type II)", style="normal", color=INK, fs=6.5)
    for x, g in ((9, "TAS1R1"), (19, "TAS1R3"), (29, "TAS1R2"), (39.5, "TAS2R1")):
        gene_node(ax, x, 45, g, sm, c)
    gene_node(ax, 39.5, 40.5, "TAS2R38", sm, c)
    note(ax, 14, 48.6, "umami"); note(ax, 24, 48.6, "sweet"); note(ax, 39.5, 48.6, "bitter")
    gene_node(ax, 25, 34, "GNAT3", sm, c)
    for x in (9, 19, 29):
        arrow(ax, (x, 43.3), (25 + (x - 19) * 0.35, 35.7))
    arrow(ax, (37, 39), (29.6, 35.2))
    gene_node(ax, 25, 27.5, "PLCB2", sm, c)
    arrow(ax, (25, 32.3), (25, 29.2))
    gene_node(ax, 25, 21, "ITPR3", sm, c)
    arrow(ax, (25, 25.8), (25, 22.7)); note(ax, 28.5, 24.3, "IP₃", ha="left")
    gene_node(ax, 25, 14.5, "TRPM5", sm, c)
    arrow(ax, (25, 19.3), (25, 16.2)); note(ax, 28.5, 17.8, "Ca$^{2+}$", ha="left")
    gene_node(ax, 15, 8.5, "CALHM1", sm, c); gene_node(ax, 35, 8.5, "CALHM3", sm, c)
    arrow(ax, (23, 12.8), (17, 10.2)); arrow(ax, (27, 12.8), (33, 10.2))
    note(ax, 25, 5.4, "ATP release to the gustatory nerve")
    ax.add_patch(FancyBboxPatch((51, 22), 46, 32, boxstyle="round,pad=0,rounding_size=8", fc="#eef4fc", ec="#c9dcf5", lw=1))
    note(ax, 74, 51.5, "Olfactory sensory neuron", style="normal", color=INK, fs=6.5)
    gene_node(ax, 59, 45, "OR7A17", sm, c); gene_node(ax, 73, 45, "TAAR5", sm, c)
    note(ax, 59, 48.6, "16 lost receptors, e.g.")
    gene_node(ax, 66, 38.5, "GNAL", sm, c)
    arrow(ax, (60, 43.3), (64.5, 40.2)); arrow(ax, (72, 43.3), (67.5, 40.2))
    gene_node(ax, 66, 32, "ADCY3", sm, c); arrow(ax, (66, 36.8), (66, 33.7))
    for x, g in ((57, "CNGA2"), (66, "CNGA4"), (75, "CNGB1")):
        gene_node(ax, x, 25.5, g, sm, c, w=8)
    arrow(ax, (66, 30.3), (66, 27.2)); note(ax, 69.5, 28.8, "cAMP", ha="left")
    gene_node(ax, 90, 25.5, "ANO2", sm, c, w=8)
    arrow(ax, (79, 25.5), (85.9, 25.5)); note(ax, 82.5, 27.7, "Ca$^{2+}$")
    ax.add_patch(FancyBboxPatch((51, 4), 46, 14, boxstyle="round,pad=0,rounding_size=6", fc="#eef4fc", ec="#c9dcf5", lw=1))
    note(ax, 74, 15.4, "Liver and blood: hydrolysis of organophosphates", style="normal", color=INK, fs=6)
    for x, g in ((61, "PON1"), (74, "PON2"), (87, "PON3")):
        gene_node(ax, x, 9.5, g, sm, c)
    map_legend(ax, c, 59.5, "dots: lost in manatee, cetaceans, pinnipeds")


def eye_map(ax, se, letter, title="Subterranean mammals: the eye"):
    from matplotlib.patches import Ellipse
    c = C3
    map_axes(ax)
    panel(ax, letter, title)
    ax.add_patch(Ellipse((50, 28), 92, 54, fc="#eaf7f1", ec="#bfe6d4", lw=1.2))
    ax.add_patch(Ellipse((16, 28), 22, 36, fc="#ffffff", ec="#bfe6d4", lw=1))
    note(ax, 16, 44.5, "Lens fibre cells", style="normal", color=INK, fs=6.5)
    note(ax, 16, 41.2, "crystallins")
    for i, g in enumerate(["CRYAA", "CRYBA1", "CRYBA4", "CRYBB2", "CRYGC", "CRYGS"]):
        gene_node(ax, 11 + (i % 2) * 10, 37.5 - (i // 2) * 4.3, g, se, c)
    note(ax, 16, 24.6, "filaments, membrane, junctions")
    for i, g in enumerate(["BFSP1", "BFSP2", "LGSN", "MIP", "GJA3", "GJA8"]):
        gene_node(ax, 11 + (i % 2) * 10, 21 - (i // 2) * 4.3, g, se, c)
    gene_node(ax, 16, 8, "LCTL", se, c)
    note(ax, 37, 42, "Vitreous", style="normal", color=INK, fs=6.5)
    gene_node(ax, 37, 38, "OPTC", se, c)
    note(ax, 70, 51.5, "Retina", style="normal", color=INK, fs=6.5)
    note(ax, 50, 48, "rod photoreceptor", style="normal", color=INK, fs=6)
    gene_node(ax, 52, 43, "RHO", se, c); gene_node(ax, 52, 36.5, "GNAT1", se, c)
    gene_node(ax, 47.5, 30, "PDE6A", se, c, w=8); gene_node(ax, 56.5, 30, "PDE6B", se, c, w=8)
    gene_node(ax, 52, 23.5, "CNGA1", se, c)
    arrow(ax, (52, 41.3), (52, 38.2)); arrow(ax, (52, 34.8), (52, 31.7)); arrow(ax, (52, 28.3), (52, 25.2), style="-[")
    note(ax, 58, 20.6, "outer-segment discs")
    gene_node(ax, 58, 17, "PRPH2", se, c); gene_node(ax, 58, 12.5, "ROM1", se, c)
    note(ax, 70, 47.6, "photoreceptor synapse")
    gene_node(ax, 70, 43.5, "MPP4", se, c)
    note(ax, 86, 39.6, "ON bipolar cell", style="normal", color=INK, fs=6)
    gene_node(ax, 84, 36, "GRM6", se, c); gene_node(ax, 84, 29, "GNAO1", se, c)
    gene_node(ax, 84, 22, "TRPM1", se, c); gene_node(ax, 84, 15, "NYX", se, c)
    arrow(ax, (84, 34.3), (84, 30.7)); arrow(ax, (84, 27.3), (84, 23.7), style="-[")
    arrow(ax, (74.5, 42.5), (79.4, 37.3)); note(ax, 79.5, 44.3, "glutamate")
    ax.add_patch(FancyBboxPatch((67.6, 19.6), 9, 15.6, boxstyle="round,pad=0,rounding_size=1", fc="none", ec=GRID,
                                lw=0.7, ls="--"))
    for y, g in ((32.5, "RGS11"), (27.5, "RGS7"), (22.5, "GNB5")):
        gene_node(ax, 72.1, y, g, se, c, w=8)
    arrow(ax, (76.7, 29), (79.4, 29), style="-[")
    gene_node(ax, 72, 14, "CABP5", se, c)
    map_legend(ax, c, 59.5, "dots: lost in each of the six subterranean origins")


def map_status():
    tree, names, Y, trait = loss_setup()
    return (gene_status("marine", MAP_MARINE_GENES, tree, names, Y, trait),
            gene_status("hypoxia_subterranean", MAP_EYE_GENES, tree, names, Y, trait))


# ---------------------------------------------------------------------------------------
# Genome-wide results: calls and calibration (used in Fig. 4)
# ---------------------------------------------------------------------------------------

def trait_summary():
    rows = []
    for key, label in TRAITS:
        s_ = json.loads((RES / f"{key}_toga" / "summary.json").read_text())
        for lay, _, _ in LAYERS:
            rows.append(dict(trait=label, layer=lay, calls=s_[f"calls_{lay}"],
                             any_call=100 * s_[f"control_sets_with_any_call_{lay}"], origins=s_["origins"]))
    return pd.DataFrame(rows)


def calls_panel(ax, d, letter):
    y = np.arange(len(TRAITS))
    h = 0.26
    for i, (lay, name, col) in enumerate(LAYERS):
        v = d[d.layer == lay].calls.to_numpy()
        ax.barh(y + (i - 1) * h, v, height=h - 0.04, color=col, label=name)
        for yy, vv in zip(y + (i - 1) * h, v):
            if vv:
                ax.text(vv + 0.8, yy, str(vv), va="center", fontsize=5, color=INK2)
    origins = d.drop_duplicates("trait").origins.tolist()
    ax.set_yticks(y, [f"{t} ({o})" for (_, t), o in zip(TRAITS, origins)])
    ax.set_ylim(len(TRAITS) - 0.5, -0.8)
    ax.set_xlim(0, 60)
    ax.set_xlabel("Genes called (q = 0.1)")
    ax.legend(handles=[Rectangle((0, 0), 1, 1, color=c, label=n + (" (no calls)" if lay == "rate" else ""))
                       for lay, n, c in LAYERS], loc="lower right")
    grid(ax, "x")
    panel(ax, letter, "Convergent genes (origins)")


def calibration_panel(ax, d, letter):
    from scipy.stats import beta as _beta
    y = np.arange(len(TRAITS))
    n = 100                                         # control trait sets per analysis
    for i, (lay, name, col) in enumerate(LAYERS):
        v = d[d.layer == lay].any_call.to_numpy()
        k = np.round(v * n / 100)
        lo = np.where(k > 0, _beta.ppf(0.025, k, n - k + 1), 0) * 100        # exact 95% interval
        hi = np.where(k < n, _beta.ppf(0.975, k + 1, n - k), 1) * 100
        ax.hlines(y + (i - 1) * 0.22, lo, hi, color=col, lw=0.7, zorder=2)
        ax.scatter(v, y + (i - 1) * 0.22, s=14, color=col, edgecolor="white", lw=0.6, zorder=3, label=name)
    ax.axvline(10, color=INK2, lw=0.6, ls="--")
    ax.set_yticks(y, [""] * len(y))
    ax.set_ylim(len(TRAITS) - 0.5, -0.8)
    ax.text(10.6, -0.62, "nominal 10%", fontsize=5, color=INK2, va="center")
    ax.set_xlim(-1.5, 32)
    ax.set_xlabel("Control traits with any call (%)")
    grid(ax, "x")
    panel(ax, letter, "Calibration, 100 controls")


# ---------------------------------------------------------------------------------------
# Nature Ecology & Evolution figure set: Figs 1-6 and Extended Data Figs 1-10
# ---------------------------------------------------------------------------------------

def nee_fig4():
    """Genome-wide results: calls, calibration, volcano plots, subterranean vs high altitude."""
    tree, names, Y, trait = loss_setup()
    d = trait_summary()
    hyp = pd.read_csv(RES / "extra" / "joint_subterranean_altitude.csv")
    t = pd.read_csv(RES / "extra" / "loss_timing.csv")
    sg = list(dict.fromkeys(t[t.trait == "hypoxia_subterranean"].gene))
    fig = plt.figure(figsize=(183 * MM, 118 * MM))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.05], hspace=0.55)
    top = gs[0].subgridspec(1, 3, width_ratios=[1.25, 0.75, 1.0], wspace=0.12)
    calls_panel(fig.add_subplot(top[0]), d, "a")
    calibration_panel(fig.add_subplot(top[1]), d, "b")
    joint_panel(fig.add_subplot(top[2]), hyp, "beta_subterranean", "beta_altitude", sg,
                ["EPAS1", "EGLN1", "HIF1A", "HBB", "NGB", "CYGB"], "c", "Subterranean vs high altitude",
                "Partial effect: subterranean", "Partial effect: high altitude")
    vs = gs[1].subgridspec(1, 4, wspace=0.32)
    for i, (k, lab, c) in enumerate(VOLCANO_TRAITS):
        ax = fig.add_subplot(vs[i])
        volcano_panel(ax, k, lab, c, Y, trait, "defg"[i], n_label=6)
        if i == 0:
            ax.set_ylabel("−log₁₀ p (loss)")
    save(fig, "Figure4")


def nee_fig5():
    """Marine: PON1 mutations on the species tree and the codon alignment."""
    from mlconv.tree import subtree
    ctx = marine_context()
    fig = plt.figure(figsize=(183 * MM, 172 * MM))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 1], hspace=0.36)
    mutation_map_into(fig, gs[0], ctx, "a")
    t1, n1 = subtree(ctx["full"], ctx["names"], list(ctx["marine"]) + ["Hippopotamus_amphibius", "Bos_taurus",
                                                                       "Canis_lupus_familiaris", "Loxodonta_africana"])
    alignment_panel(fig.add_subplot(gs[1]), "PON1", ((84, 96), (217, 229)), n1, ctx["common"], ctx["marine"], C1, "b",
                    "PON1 codons 84–96 and 217–229: frameshifts shared by all cetaceans (dashed)", shared=(91, 224))
    save(fig, "Figure5")


def nee_fig6():
    """Subterranean: the eye pathway map, LGSN alignment, enrichment."""
    import alignment_detail as AD
    from mlconv.tree import subtree
    tree, names, Y, trait = loss_setup()
    common = pd.read_csv(RES / "toga" / "assemblies_paper.tsv", sep="\t").set_index("species")["Common name"].to_dict()
    common.update(SHORT_NAMES)
    common.update(DETAIL_NAMES)
    subt = set(np.array(names)[trait["hypoxia_subterranean"]])
    t2, n2 = subtree(tree, names, list(subt) + AD.SUBTERRANEAN_RELATIVES)
    _, se = map_status()
    tis = pd.read_csv(RES / "extra" / "enrichment_tissue.csv")
    go = pd.read_csv(RES / "extra" / "enrichment_go.csv")
    fig = plt.figure(figsize=(183 * MM, 168 * MM))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.45, 1], hspace=0.12)
    eye_map(fig.add_subplot(gs[0]), se, "a")
    bot = gs[1].subgridspec(1, 2, width_ratios=[1.9, 1], wspace=0.3)
    alignment_panel(fig.add_subplot(bot[0]), "LGSN", ((186, 189), (375, 386)), n2, common, subt, C3, "b",
                    "LGSN codons 186\u2013189 and 375\u2013386", shared=(376,))
    en = bot[1].subgridspec(2, 1, hspace=0.9)
    k = "hypoxia_subterranean"
    enrichment_bars(fig.add_subplot(en[0]), tis[tis.trait == k], C3, "c", "Tissue (Human Protein Atlas)", n=2)
    enrichment_bars(fig.add_subplot(en[1]), go[go.trait == k], C3, "d", "GO biological process", n=3, max_size=450,
                    rename={"lens development in camera-type eye": "lens development"})
    save(fig, "Figure6")


def ed_heatmaps():
    tree, names = from_newick((RES / "toga" / "master_paper.nwk").read_text())
    status = pd.read_csv(RES / "toga" / "loss_status_paper.tsv.gz", sep="\t", index_col=0)
    mar = pd.read_csv(RES / "marine_toga" / "marine_genes.csv")
    sub = pd.read_csv(RES / "hypoxia_subterranean_toga" / "hypoxia_subterranean_genes.csv")
    mar_genes = mar[mar.call_loss == True].sort_values("p_loss").symbol.tolist()  # noqa: E712
    sub_genes = sub[sub.call_loss == True].sort_values("p_loss").symbol.tolist()  # noqa: E712
    mar_species = (RES / "marine_toga" / "trait_species.txt").read_text().split()
    sub_species = (RES / "hypoxia_subterranean_toga" / "trait_species.txt").read_text().split()
    fig = plt.figure(figsize=(183 * MM, 176 * MM))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.4, 1.1], hspace=0.2)
    loss_panel(fig.add_subplot(gs[0]), "marine", mar_genes, f"Marine mammals: {len(mar_genes)} convergent losses",
               "a", tree, names, status, mar_species, C5)
    loss_panel(fig.add_subplot(gs[1]), "sub", sub_genes, f"Subterranean mammals: {len(sub_genes)} convergent losses",
               "b", tree, names, status, sub_species, C5)
    handles = [Rectangle((0, 0), 1, 1, color=INK, label="lost (TOGA L or UL)"),
               Rectangle((0, 0), 1, 1, color="#e8e7e3", label="intact (I or PI)"),
               Rectangle((0, 0), 1, 1, fc="white", ec=MUTED, lw=0.5, label="missing"),
               Rectangle((0, 0), 1, 1, color=C5, label="trait species")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    save(fig, "ExtendedDataFigure4")


def ed_marine_lineages():
    ctx = marine_context()
    fig = plt.figure(figsize=(183 * MM, 120 * MM))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.6, 1], hspace=0.45)
    funnels_into(fig, gs[0], ctx)
    synteny_into(fig, gs[1], ctx, "d")
    save(fig, "ExtendedDataFigure7")


def ed_taste_map():
    sm, _ = map_status()
    fig = plt.figure(figsize=(183 * MM, 118 * MM))
    taste_map(fig.add_subplot(111), sm, "a")
    save(fig, "ExtendedDataFigure6")


# ---------------------------------------------------------------------------------------
# Study summary: species tree with the convergent losses, workflow and main results
# ---------------------------------------------------------------------------------------

SILHOUETTES = [  # (group, icon, colour, where): None = group centre; a trait key = that group's trait species;
    # "not:<trait>" = the group's other species; a tuple of name prefixes = those species
    ("Bats", "bat", C2, "echolocation"), ("Whales and dolphins", "dolphin", C1, "marine"),
    ("Camelids", "camel", C4, "hypoxia_altitude"), ("Ruminants, pigs", "bison", C4, "hypoxia_altitude"),
    ("Carnivores", "dog", INK2, "not:marine"), ("Carnivores", "sealion", C1, "marine"),
    ("Xenarthrans", "anteater", INK2, None), ("Moles, hedgehog", "mole", C3, "hypoxia_subterranean"),
    ("Primates", "gorilla", INK2, None), ("Rodents", "mouse", INK2, "not:hypoxia_subterranean"),
    ("Rodents", "mouse", C3, ("Heterocephalus_", "Fukomys_")), ("Rabbits, pika", "rabbit", C4, "hypoxia_altitude"),
    ("Afrotherians", "elephant", INK2, None)]


def place_silhouettes(fig, ax, lineage, group, th, r, size, names):
    """Silhouettes (game-icons.net, CC BY 3.0; scripts/silhouettes.json) at radius r of a polar tree axis. Grey ones
    sit at the centre of their group; coloured ones at the mean angle of the group's species with that trait."""
    import convergence_toga as CT
    icons = json.loads((ROOT / "scripts" / "silhouettes.json").read_text())["icons"]
    n = len(th)
    grp = np.array(group[:n])
    has = {k: np.array([CT.TRAITS[k][0](names[i], lineage.iloc[i]) for i in range(n)])
           for k in ("echolocation", "marine", "hypoxia_subterranean", "hypoxia_altitude")}
    fig.canvas.draw()
    fw, fh = fig.get_size_inches()
    for g, key, col, where in SILHOUETTES:
        sel = grp == g
        if isinstance(where, tuple):
            sel = np.array([any(nm.startswith(k) for k in where) for nm in names[:n]])
        elif isinstance(where, str):
            sel = sel & (~has[where[4:]] if where.startswith("not:") else has[where])
        if not sel.any():
            continue
        px, py = fig.transFigure.inverted().transform(ax.transData.transform((th[sel].mean(), r)))
        w, h = size, size * fw / fh
        iax = fig.add_axes([px - w / 2, py - h / 2, w, h])
        ic = icons[key]
        iax.add_patch(PathPatch(MPath(np.array(ic["vertices"]), np.array(ic["codes"], dtype=np.uint8)), fc=col, lw=0))
        iax.set_xlim(0, ic["w"])
        iax.set_ylim(ic["h"], 0)
        iax.set_aspect("equal")
        iax.set_axis_off()


def summary_box(ax, x, y, w, h, text, col, fs=6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.015", fc=PALE, ec=col, lw=0.9))
    ax.text(x + 0.03, y + h / 2, text, fontsize=fs, color=INK, va="center", ha="left", linespacing=1.25)


def study_summary():
    import convergence_toga as CT
    from mlconv.tree import origin_nodes as onodes
    tree, names = from_newick((RES / "toga" / "master_paper.nwk").read_text())
    cov = pd.read_csv(RES / "toga" / "coverage_paper.tsv", sep="\t").drop_duplicates("species").set_index("species")
    lineage = cov.lineage.reindex(names).fillna("")
    lineage.iloc[names.index("Homo_sapiens")] = "Eutheria; Euarchontoglires; Primates; Haplorrhini"
    _, _, Y, _ = loss_setup()
    calls = pd.read_csv(RES / "extra" / "supplementary_table1_convergent_losses.csv")
    lost = {t: Y.reindex(calls[calls.trait == t].gene).reindex(columns=names).eq(1).sum().to_numpy()
            for t in ("Marine", "Subterranean")}
    n = tree.n_tips
    _, y = tree_xy(tree)
    height = np.zeros(tree.n_nodes)
    for v in range(tree.n_nodes - 1):
        height[tree.parent[v]] = max(height[tree.parent[v]], height[v] + 1)
    x = 1 - height / height.max()
    span = 2 * np.pi * 0.965
    theta = lambda v: np.pi / 2 - y[v] / (n - 1) * span          # noqa: E731
    width = span / (n - 1)

    fig = plt.figure(figsize=(183 * MM, 150 * MM))
    ax = fig.add_axes([0.005, 0.085, 0.575, 0.87], projection="polar")
    ax.set_axis_off()
    for v in range(tree.n_nodes - 1):
        ax.plot([theta(v), theta(v)], [x[tree.parent[v]], x[v]], color=MUTED, lw=0.6, solid_capstyle="butt")
    for v in range(n, tree.n_nodes):
        ch = np.nonzero(tree.parent == v)[0]
        a = np.linspace(min(theta(c) for c in ch), max(theta(c) for c in ch), 30)
        ax.plot(a, np.full_like(a, x[v]), color=MUTED, lw=0.6)
    th = np.array([theta(t) for t in range(n)])
    ring0, ring_w = 1.03, 0.045
    for k, (key, label, col) in enumerate(TREE_TRAITS):
        tr = np.array([CT.TRAITS[key][0](s_, l) for s_, l in zip(names, lineage)])
        ax.bar(th, ring_w, width=width * 0.98, bottom=ring0 + k * (ring_w + 0.006), color=np.where(tr, col, PALE), lw=0)
        for o in onodes(tree, tr):
            ax.scatter([theta(o)], [(x[tree.parent[o]] + x[o]) / 2], s=16, color=col, edgecolor="white", lw=0.6,
                       zorder=5)
    # outer ring: how many of the convergent losses each species carries
    base = ring0 + 4 * (ring_w + 0.006) + 0.03
    scale = 0.42 / 48
    ax.bar(th, lost["Marine"] * scale, width=width * 0.9, bottom=base, color=C1, lw=0)
    ax.bar(th, lost["Subterranean"] * scale, width=width * 0.9, bottom=base + lost["Marine"] * scale, color=C3, lw=0)
    ref = np.linspace(th.min() - width / 2, th.max() + width / 2, 200)
    for k in (10, 20, 30, 40):
        ax.plot(ref, np.full_like(ref, base + k * scale), color=GRID, lw=0.4, zorder=0)
    ang0 = np.pi / 2 + 0.035 * np.pi
    for k in (0, 20, 40):
        ax.text(ang0, base + k * scale, str(k), fontsize=5, color=INK2, ha="center", va="center")
    # order arcs
    group = [next((lab for key, lab in ORDERS if key in l), "") for l in lineage]
    ro = base + 0.5
    start = 0
    for t in range(1, n + 1):
        if t == n or group[t] != group[start]:
            if group[start]:
                a = np.linspace(theta(start) + width * 0.4, theta(t - 1) - width * 0.4, 40)
                ax.plot(a, np.full_like(a, ro), color=INK, lw=1.0)
                if t - start >= 4:
                    mid = (theta(start) + theta(t - 1)) / 2
                    deg = np.degrees(mid) % 360
                    flip = 180 < deg < 360
                    lab = group[start]
                    ax.text(mid, ro + 0.04, lab, fontsize=5.5, color=INK, ha="center", va="top" if flip else "bottom",
                            rotation=deg - 90 + (180 if flip else 0), rotation_mode="anchor", linespacing=1.0)
            start = t
    ax.set_ylim(0, ro + 0.55)
    place_silhouettes(fig, ax, lineage, group, th, ro + 0.4, 0.046, names)
    fig.text(0.015, 0.975, "a", fontsize=8, fontweight="bold", va="top")
    fig.text(0.04, 0.975, "115 mammals, 15,703 genes, four traits", fontsize=7, va="top")
    handles = [Rectangle((0, 0), 1, 1, color=c, label=f"{lab} ({len(onodes(tree, np.array([CT.TRAITS[k][0](s_, l) for s_, l in zip(names, lineage)])))} origins)")
               for k, lab, c in TREE_TRAITS]
    fig.legend(handles=handles, loc="lower left", ncol=2, bbox_to_anchor=(0.02, 0.0), fontsize=6, handlelength=1.2,
               title="Inner rings: trait species; dots: independent origins", title_fontsize=6)
    fig.legend(handles=[Rectangle((0, 0), 1, 1, color=C1, label="of the 48 marine losses"),
                        Rectangle((0, 0), 1, 1, color=C3, label="of the 15 subterranean losses")],
               loc="lower left", bbox_to_anchor=(0.365, 0.0), fontsize=6, handlelength=1.2,
               title="Outer bars: genes lost per species", title_fontsize=6)

    # b: workflow
    axb = fig.add_axes([0.6, 0.6, 0.39, 0.37])
    axb.set_axis_off()
    axb.set_xlim(0, 1)
    axb.set_ylim(0, 1)
    steps = [("Zoonomia alignment, TOGA orthology: 115 species", INK2),
             ("Four layers per gene: nucleotide rate, protein rate,\ntheir contrast, gene loss", C1),
             ("Null learned from all genes (GLS); gene-loss null\nsimulated with species × gene annotation error", C1),
             ("Signal must survive removal of each origin", C1),
             ("Repeated on 20 matched control traits", C6)]
    hgt, gap = 0.135, 0.045
    for i, (txt, col) in enumerate(steps):
        y0 = 0.86 - i * (hgt + gap) - hgt
        summary_box(axb, 0.0, y0, 1.0, hgt, txt, col)
        if i:
            axb.annotate("", (0.5, y0 + hgt), (0.5, y0 + hgt + gap), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.6,
                                                                                     shrinkA=0, shrinkB=0))
    axb.text(0.0, 0.98, "b", fontsize=8, fontweight="bold", va="top")
    axb.text(0.07, 0.98, "Calibrated convergence test", fontsize=7, va="top")

    # c: results per trait
    axc = fig.add_axes([0.71, 0.35, 0.16, 0.16])
    rows = [("Marine", 48, C1, "smell, taste, skin"), ("Underground", 15, C3, "lens, retina"),
            ("High altitude", 1, C4, "likely error"), ("Echolocation", 0, C2, "")]
    yy = np.arange(len(rows))[::-1]
    axc.barh(yy, [r[1] for r in rows], color=[r[2] for r in rows], height=0.62)
    for yv, (lab, v, c, note_) in zip(yy, rows):
        axc.text(v + 1.5, yv, f"{v}  {note_}" if v else "0", fontsize=5.5, color=INK2, va="center")
    axc.set_yticks(yy, [r[0] for r in rows])
    axc.set_xlim(0, 50)
    axc.set_xlabel("Convergent gene losses (q < 0.1)")
    axc.spines["left"].set_visible(False)
    axc.tick_params(axis="y", length=0)
    axc.text(-0.68, 1.16, "c", transform=axc.transAxes, fontsize=8, fontweight="bold", va="bottom")
    axc.text(-0.56, 1.16, "Convergence detected only as gene loss", transform=axc.transAxes, fontsize=7, va="bottom")
    axc.text(5, yy[-1], "no convergent rate shift in any trait", fontsize=5.5, color=INK2, va="center")

    # d: calibration
    axd = fig.add_axes([0.71, 0.1, 0.27, 0.14])
    cal = [("Rate test, ours", 0.000, 0.007, C6), ("Rate tests, others", 0.19, 0.91, MUTED),
           ("Loss test, ours", 0.0, 0.067, C6), ("Loss, no phylogeny", 0.0, 0.69, MUTED)]
    yy = np.arange(len(cal))[::-1]
    for yv, (lab, lo, hi, c) in zip(yy, cal):
        axd.plot([lo, hi], [yv, yv], color=c, lw=3, solid_capstyle="round")
    axd.axvline(0.1, color=INK2, lw=0.7, ls="--")
    axd.set_yticks(yy, [c[0] for c in cal])
    axd.set_xlim(-0.02, 1)
    axd.set_xlabel("False discovery rate, simulations")
    axd.spines["left"].set_visible(False)
    axd.tick_params(axis="y", length=0)
    axd.text(-0.55, 1.16, "d", transform=axd.transAxes, fontsize=8, fontweight="bold", va="bottom")
    axd.text(-0.45, 1.16, "False calls are controlled", transform=axd.transAxes, fontsize=7, va="bottom")
    axd.text(0, -0.48, "Real genomes: 13% of 500 control traits had any loss call", transform=axd.transAxes,
             fontsize=5.5, color=INK2, va="top")
    fig.text(0.985, 0.004, "Silhouettes: game-icons.net (CC BY 3.0)", fontsize=5, color=MUTED, va="bottom", ha="right")
    save(fig, "StudySummary")


if __name__ == "__main__":
    figure_tree()          # Fig. 1
    figure1()              # Fig. 2
    figure3()              # Fig. 3
    nee_fig4()
    nee_fig5()
    nee_fig6()
    figure2()              # Extended Data Fig. 1
    supp1()                # Extended Data Fig. 2
    ed_sensitivity()       # Extended Data Fig. 3
    ed_heatmaps()          # Extended Data Fig. 4
    figure5()              # Extended Data Fig. 8
    ed_marine_lineages()   # Extended Data Fig. 7
    ed_taste_map()         # Extended Data Fig. 6
    figure9()              # Extended Data Fig. 5
    supp2()                # Extended Data Fig. 10
    supp3()                # Extended Data Fig. 9
    study_summary()        # one-page overview (talks, cover letter, preprint)
