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


def kingman_tree(n_tips: int, rng: np.random.Generator) -> Tree:
    """Random ultrametric tree from the Kingman coalescent (heights in N units)."""
    n_nodes = 2 * n_tips - 1
    parent = -np.ones(n_nodes, dtype=int)
    height = np.zeros(n_nodes)
    active = list(range(n_tips))
    t = 0.0
    nxt = n_tips
    while len(active) > 1:
        k = len(active)
        t += rng.exponential(1.0 / (k * (k - 1) / 2))
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
