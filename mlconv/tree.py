"""Minimal phylogeny utilities: random trees, foreground mapping, permulations.

Nodes are indexed so that every child has a smaller index than its parent:
tips are 0..n-1, internal nodes n..2n-2, and the root is 2n-2. Each non-root
node owns the branch leading to it, so branch b == node b for b < root.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Tree:
    parent: np.ndarray        # parent[node]; -1 for the root
    length: np.ndarray        # branch length above each node (0 for root)
    n_tips: int

    @property
    def n_nodes(self) -> int:
        return len(self.parent)

    @property
    def root(self) -> int:
        return self.n_nodes - 1

    @property
    def n_branches(self) -> int:
        return self.n_nodes - 1

    def tip_sets(self) -> np.ndarray:
        """Boolean matrix D[node, tip] = tip descends from node."""
        D = np.zeros((self.n_nodes, self.n_tips), dtype=bool)
        D[np.arange(self.n_tips), np.arange(self.n_tips)] = True
        for child in range(self.n_nodes - 1):
            D[self.parent[child]] |= D[child]
        return D

    def path_matrix(self) -> np.ndarray:
        """A[b, a] = 1 if branch a lies on the path root -> b (inclusive)."""
        A = np.eye(self.n_branches)
        for b in range(self.n_branches - 1, -1, -1):
            p = self.parent[b]
            if p != self.root:
                A[b] += A[p]
        return A


def kingman_tree(n_tips: int, rng: np.random.Generator, process: str = "coalescent") -> Tree:
    """Random ultrametric tree, normalised to height 1.

    process="coalescent": Kingman coalescent (long internal branches near the root).
    process="yule": pure-birth tree (waiting time with k lineages ~ Exp(k)),
    closer in shape to species phylogenies.
    """
    n_nodes = 2 * n_tips - 1
    parent = -np.ones(n_nodes, dtype=int)
    height = np.zeros(n_nodes)
    active = list(range(n_tips))
    t = 0.0
    nxt = n_tips
    while len(active) > 1:
        k = len(active)
        rate = k * (k - 1) / 2 if process == "coalescent" else k
        t += rng.exponential(1.0 / rate)
        i, j = rng.choice(k, size=2, replace=False)
        a, b = active[i], active[j]
        parent[a] = parent[b] = nxt
        height[nxt] = t
        active = [x for x in active if x not in (a, b)] + [nxt]
        nxt += 1
    length = np.zeros(n_nodes)
    for node in range(n_nodes - 1):
        length[node] = height[parent[node]] - height[node]
    return Tree(parent=parent, length=length / height[-1], n_tips=n_tips)


def foreground_branches(tree: Tree, tip_trait: np.ndarray, D: np.ndarray | None = None) -> np.ndarray:
    """RERconverge-style mapping: a branch is foreground when every tip below it has the trait."""
    D = tree.tip_sets() if D is None else D
    fg = np.array([tip_trait[D[b]].all() for b in range(tree.n_branches)])
    return fg


def pick_convergent_clades(tree: Tree, n_origins: int, size_range: tuple[int, int],
                           rng: np.random.Generator) -> np.ndarray:
    """Choose non-overlapping clades as independent trait origins; return tip trait vector."""
    D = tree.tip_sets()
    sizes = D.sum(1)
    candidates = [v for v in range(tree.n_tips, tree.root)
                  if size_range[0] <= sizes[v] <= size_range[1]]
    rng.shuffle(candidates)
    trait = np.zeros(tree.n_tips, dtype=bool)
    chosen = 0
    for v in candidates:
        if not (trait & D[v]).any():
            trait |= D[v]
            chosen += 1
            if chosen == n_origins:
                break
    if chosen < n_origins:
        raise ValueError("could not place the requested number of origins")
    return trait


def clade_permute(tree: Tree, origin_sizes: list[int], n_perm: int, rng: np.random.Generator,
                  exclude: np.ndarray | None = None, size_tol: int = 1,
                  max_tries: int = 1000) -> np.ndarray:
    """Structure-matched null foregrounds drawn from control lineages.

    Each null draw places the same number of independent origins as observed,
    each a random clade within `size_tol` tips of an observed origin, with no
    overlap. Clades containing any `exclude` tip (normally the trait-positive
    tips) are never used, following the control-lineage logic of Thomas & Hahn
    (2015): on a finite tree the null would otherwise keep re-drawing the true
    trait clades and swallow the signal.
    """
    D = tree.tip_sets()
    sizes = D.sum(1)
    allowed = np.ones(tree.root, dtype=bool) if exclude is None else ~(D[: tree.root] & exclude).any(1)
    by_size = {k: [v for v in range(tree.root) if allowed[v] and abs(sizes[v] - k) <= size_tol]
               for k in set(origin_sizes)}
    out = np.zeros((n_perm, tree.n_branches), dtype=bool)
    for i in range(n_perm):
        for _ in range(max_tries):
            trait = np.zeros(tree.n_tips, dtype=bool)
            ok = True
            for k in origin_sizes:
                v = by_size[k][rng.integers(len(by_size[k]))]
                if (trait & D[v]).any():
                    ok = False
                    break
                trait |= D[v]
            if ok:
                break
        out[i] = foreground_branches(tree, trait, D)
    return out


def node_heights(tree: Tree) -> np.ndarray:
    """Distance from each node to its deepest descendant tip."""
    h = np.zeros(tree.n_nodes)
    for c in range(tree.n_nodes - 1):
        p = tree.parent[c]
        h[p] = max(h[p], h[c] + tree.length[c])
    return h


def origin_nodes(tree: Tree, tip_trait: np.ndarray) -> list[int]:
    """Top node (branch) of each maximal trait-positive clade, i.e. each independent origin."""
    fg = foreground_branches(tree, tip_trait)
    return [b for b in range(tree.n_branches)
            if fg[b] and (tree.parent[b] == tree.root or not fg[tree.parent[b]])]


def origin_index(tree: Tree, tip_trait: np.ndarray) -> np.ndarray:
    """(n_branches,) index of the independent origin each foreground branch belongs to; -1 otherwise."""
    D = tree.tip_sets()
    out = -np.ones(tree.n_branches, dtype=int)
    for i, o in enumerate(origin_nodes(tree, tip_trait)):
        inside = ~(D[: tree.n_branches] & ~D[o]).any(1)
        out[inside] = i
    return out


def matched_clade_permute(tree: Tree, origins: list[int], n_perm: int, rng: np.random.Generator,
                          exclude: np.ndarray | None = None, size_tol: int = 1,
                          age_ratio: float = 1.5, min_candidates: int = 5,
                          max_tries: int = 1000) -> np.ndarray:
    """Control-lineage null foregrounds matched on number of origins, clade size and clade age.

    For each observed origin with k tips and crown-plus-stem age a (the height of its
    parent node), candidates are control clades with |size - k| <= size_tol and age
    within a factor `age_ratio`. Matching age matters because young clades are made of
    short branches, whose estimated rates are noisier; a size-only null then
    understates the variance of the observed foreground. If fewer than
    `min_candidates` qualify, the age window is widened stepwise.
    """
    D = tree.tip_sets()
    sizes = D.sum(1)
    h = node_heights(tree)
    stem_age = np.array([h[tree.parent[v]] for v in range(tree.root)])
    allowed = np.ones(tree.root, dtype=bool) if exclude is None else ~(D[: tree.root] & exclude).any(1)
    pools = []
    for o in origins:
        r = age_ratio
        while True:
            ok = (allowed & (np.abs(sizes[: tree.root] - sizes[o]) <= size_tol)
                  & (stem_age >= stem_age[o] / r) & (stem_age <= stem_age[o] * r))
            if ok.sum() >= min_candidates or r > 64:
                break
            r *= 1.5
        pools.append(np.nonzero(ok)[0])
    out = np.zeros((n_perm, tree.n_branches), dtype=bool)
    for i in range(n_perm):
        for _ in range(max_tries):
            trait = np.zeros(tree.n_tips, dtype=bool)
            ok = True
            for pool in pools:
                v = pool[rng.integers(len(pool))]
                if (trait & D[v]).any():
                    ok = False
                    break
                trait |= D[v]
            if ok:
                break
        out[i] = foreground_branches(tree, trait, D)
    return out


def origin_sizes(tree: Tree, tip_trait: np.ndarray) -> list[int]:
    """Tip counts of the maximal trait-positive clades (one per independent origin)."""
    D = tree.tip_sets()
    fg = foreground_branches(tree, tip_trait, D)
    tops = [b for b in range(tree.n_branches)
            if fg[b] and (tree.parent[b] == tree.root or not fg[tree.parent[b]])]
    return sorted(int(D[b].sum()) for b in tops)


def permulate(tree: Tree, observed_fg: np.ndarray, n_perm: int, rng: np.random.Generator,
              max_tries: int = 200, tol: int = 1) -> np.ndarray:
    """Phylogenetic permulations for a binary trait.

    Simulate Brownian motion on the tree, label the top-k tips as trait-positive
    (k = observed number of positive tips), and keep the draw if the number of
    foreground branches matches the observed number within `tol`. The null
    foregrounds therefore share the phylogenetic clumping of the real one.
    Returns a boolean matrix (n_perm, n_branches).
    """
    D = tree.tip_sets()
    A = tree.path_matrix()
    k = int(observed_fg[: tree.n_tips].sum())
    target = int(observed_fg.sum())
    sd = np.sqrt(tree.length[: tree.n_branches])
    out = np.zeros((n_perm, tree.n_branches), dtype=bool)
    filled = 0
    tries = 0
    while filled < n_perm:
        tries += 1
        values = A @ (rng.standard_normal(tree.n_branches) * sd)
        tips = values[: tree.n_tips]
        trait = np.zeros(tree.n_tips, dtype=bool)
        trait[np.argsort(tips)[-k:]] = True
        fg = foreground_branches(tree, trait, D)
        if abs(int(fg.sum()) - target) <= tol or tries > max_tries * n_perm:
            out[filled] = fg
            filled += 1
    return out


def yule_tree(n_tips: int, rng: np.random.Generator) -> Tree:
    return kingman_tree(n_tips, rng, process="yule")


def to_newick(tree: Tree, scale: float = 1.0, lengths: np.ndarray | None = None,
              annotations: dict[int, str] | None = None) -> str:
    """Newick string with tips named t0..t{n-1}; optional per-branch lengths and
    per-branch comment annotations (e.g. AliSim branch-specific models)."""
    lengths = tree.length if lengths is None else lengths
    annotations = annotations or {}
    children = [[] for _ in range(tree.n_nodes)]
    for c in range(tree.n_nodes - 1):
        children[tree.parent[c]].append(c)

    def rec(v: int) -> str:
        s = f"t{v}" if v < tree.n_tips else "(" + ",".join(rec(c) for c in children[v]) + ")"
        if v != tree.root:
            s += f":{lengths[v] * scale:.8f}" + annotations.get(v, "")
        return s

    return rec(tree.root) + ";"


def parse_newick_splits(newick: str, n_tips: int) -> dict[frozenset, float]:
    """Branch lengths of an (unrooted or rooted) Newick tree keyed by canonical split:
    the set of tip indices on the side that does not contain tip t0."""
    pos = 0
    out: dict[frozenset, float] = {}
    everything = frozenset(range(n_tips))

    def canon(tips: frozenset) -> frozenset:
        return everything - tips if 0 in tips else tips

    def node() -> frozenset:
        nonlocal pos
        if newick[pos] == "(":
            pos += 1
            tips = frozenset()
            while True:
                tips |= node()
                if newick[pos] == ",":
                    pos += 1
                    continue
                pos += 1          # ')'
                break
        else:
            start = pos
            while newick[pos] not in ",):;":
                pos += 1
            tips = frozenset([int(newick[start:pos].strip()[1:])])
        while newick[pos] not in ":,);":
            pos += 1               # internal node labels
        if newick[pos] == ":":
            pos += 1
            start = pos
            while newick[pos] not in ",);":
                pos += 1
            key = canon(tips)
            if 0 < len(key) < n_tips - 1 or len(tips) == 1:
                out[key] = out.get(key, 0.0) + float(newick[start:pos])
        return tips

    node()
    return out


def branch_splits(tree: Tree) -> list[frozenset]:
    """Canonical split for every branch of the rooted tree (root children share one)."""
    D = tree.tip_sets()
    everything = frozenset(range(tree.n_tips))
    out = []
    for b in range(tree.n_branches):
        tips = frozenset(np.nonzero(D[b])[0].tolist())
        out.append(everything - tips if 0 in tips else tips)
    return out


def from_newick(newick: str) -> tuple[Tree, list[str]]:
    """Parse a rooted Newick tree (multifurcations allowed) into a Tree.

    Tips are numbered 0..n-1 in the order they appear; internal nodes are numbered in
    post-order, so every child has a smaller index than its parent and the root is last.
    Returns the tree and the tip names in index order.
    """
    s = "".join(newick.split()).rstrip(";")             # whitespace and line breaks are not significant
    pos = 0
    names: list[str] = []
    edges: list[tuple[object, float]] = []     # (node key, branch length) recorded per node

    def parse() -> tuple[object, float]:
        nonlocal pos
        if s[pos] == "(":
            pos += 1
            kids = []
            while True:
                kids.append(parse())
                if s[pos] == ",":
                    pos += 1
                    continue
                pos += 1                       # ')'
                break
            node = ("internal", kids)
        else:
            start = pos
            while s[pos] not in ",):":
                pos += 1
            names.append(s[start:pos].strip())
            node = ("tip", len(names) - 1)
        while pos < len(s) and s[pos] not in ":,)":
            pos += 1                           # internal node label
        length = 0.0
        if pos < len(s) and s[pos] == ":":
            pos += 1
            start = pos
            while pos < len(s) and s[pos] not in ",)":
                pos += 1
            length = float(s[start:pos])
        return node, length

    root, _ = parse()
    n_tips = len(names)
    parent: list[int] = [-1] * n_tips
    length: list[float] = [0.0] * n_tips

    def number(node) -> int:
        kind, payload = node
        if kind == "tip":
            return payload
        child_ids = [(number(child), bl) for child, bl in payload]
        idx = len(parent)
        parent.append(-1)
        length.append(0.0)
        for cid, bl in child_ids:
            parent[cid] = idx
            length[cid] = bl
        return idx

    number(root)
    return Tree(parent=np.array(parent), length=np.array(length), n_tips=n_tips), names


def named_splits(newick: str, index: dict[str, int], n_tips: int) -> dict[frozenset, float]:
    """Branch lengths keyed by canonical split, for a Newick tree whose tips are named."""
    tree, names = from_newick(newick)
    D = tree.tip_sets()
    everything = frozenset(range(n_tips))
    out: dict[frozenset, float] = {}
    for b in range(tree.n_branches):
        tips = frozenset(index[names[t]] for t in np.nonzero(D[b])[0])
        key = everything - tips if 0 in tips else tips
        out[key] = out.get(key, 0.0) + float(tree.length[b])
    return out


def prune_newick(tree: Tree, names: list[str], keep: set[str], scale: float = 1.0) -> str:
    """Newick of `tree` restricted to the tips in `keep`; nodes left with one child are
    collapsed and their branch lengths summed. Tip labels are the names themselves."""
    children = [[] for _ in range(tree.n_nodes)]
    for c in range(tree.n_nodes - 1):
        children[tree.parent[c]].append(c)

    def rec(v: int):
        if v < tree.n_tips:
            return (names[v], tree.length[v]) if names[v] in keep else None
        parts = [p for p in (rec(c) for c in children[v]) if p is not None]
        if not parts:
            return None
        if len(parts) == 1:                                   # collapse unary node
            sub, bl = parts[0]
            return sub, bl + tree.length[v]
        sub = "(" + ",".join(f"{s}:{bl * scale:.8f}" for s, bl in parts) + ")"
        return sub, tree.length[v]

    out = rec(tree.root)
    return (out[0] if out[0].startswith("(") else f"({out[0]})") + ";"


def subtree(tree: Tree, names: list[str], keep: list[str]) -> tuple[Tree, list[str]]:
    """The species tree restricted to `keep` (as a Tree, with branch lengths)."""
    return from_newick(prune_newick(tree, names, set(keep)))
