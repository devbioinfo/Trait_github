"""Simulation null for convergent gene loss.

Model (per gene g, on the species tree with its branch lengths):
  * the gene is present at the root; along a branch of length t it is lost with
    probability 1 - exp(-lam_g t) and never regained (Dollo);
  * an intact gene in species s is reported lost with probability e_s (assembly gaps,
    annotation errors: a species-level false-loss rate shared by all genes);
  * the observed missing-data pattern of the gene is kept as it is.
lam_g is the maximum-likelihood rate on a log grid (pruning algorithm, all genes at once);
e_s is estimated from losses seen in one species only. Data simulated from the fitted
model contain no association with any trait but keep each gene's loss rate, the tree,
species-specific noise and missingness, so the test statistic computed on them is a null
that does not depend on how many distinct control trait sets the tree offers.

Small p-values are extrapolated with a generalized Pareto tail (Knijnenburg et al. 2009)
when fewer than 10 null values exceed the observed statistic.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as st

from .tree import Tree

FALSE_INTACT = 1e-3     # probability that a lost gene is reported intact


def false_loss_rates(Y: np.ndarray, floor: float = 1e-4) -> np.ndarray:
    """Per-species rate of losses seen in that species only (singletons), among genes
    observed in the species with at most one loss in total. Y: (genes, species) 1/0/NaN."""
    n_lost = np.nansum(Y, 1)
    rare = n_lost <= 1
    obs = ~np.isnan(Y[rare])
    single = (Y[rare] == 1)
    return np.maximum(single.sum(0) / np.maximum(obs.sum(0), 1), floor)


def loss_loglik(tree: Tree, Y: np.ndarray, e: np.ndarray, lams: np.ndarray) -> np.ndarray:
    """Log-likelihood (genes, grid) of the irreversible-loss model by pruning."""
    G, N = Y.shape
    K = len(lams)
    lik = {}
    logscale = np.zeros((G, K))
    for t in range(N):
        y = Y[:, t]
        l0 = np.where(np.isnan(y), 1.0, np.where(y == 1, e[t], 1 - e[t]))
        l1 = np.where(np.isnan(y), 1.0, np.where(y == 1, 1 - FALSE_INTACT, FALSE_INTACT))
        lik[t] = np.stack([np.repeat(l0[:, None], K, 1), np.repeat(l1[:, None], K, 1)], -1)
    for node in range(N, tree.n_nodes):
        acc = np.ones((G, K, 2))
        for c in np.nonzero(tree.parent == node)[0]:
            q = 1 - np.exp(-lams * tree.length[c])                     # (K,)
            Lc = lik.pop(c)
            acc[..., 0] *= (1 - q) * Lc[..., 0] + q * Lc[..., 1]
            acc[..., 1] *= Lc[..., 1]
        s = acc.max(-1)
        logscale += np.log(s)
        lik[node] = acc / s[..., None]
    return logscale + np.log(lik[tree.root][..., 0])                 # present at the root


def fit_loss_rates(tree: Tree, Y: np.ndarray, e: np.ndarray, n_grid: int = 60) -> np.ndarray:
    """Per-gene ML loss rate on a log grid spanning 0.01 to 100 losses per tree length."""
    T = tree.length.sum()
    lams = np.logspace(np.log10(0.01 / T), np.log10(100 / T), n_grid)
    return lams[loss_loglik(tree, Y, e, lams).argmax(1)]


def simulate_losses(tree: Tree, lam: np.ndarray, e: np.ndarray, missing: np.ndarray,
                    rng: np.random.Generator) -> np.ndarray:
    """One replicate (genes, species): 1 lost, 0 intact, NaN where the real gene is missing."""
    G = len(lam)
    state = np.zeros((G, tree.n_nodes), dtype=bool)
    for node in range(tree.n_nodes - 2, -1, -1):                   # parents before children
        p = tree.parent[node]
        q = 1 - np.exp(-lam * tree.length[node])
        state[:, node] = state[:, p] | (rng.random(G) < q)
    true = state[:, : tree.n_tips]
    u = rng.random(true.shape)
    obs = np.where(true, u >= FALSE_INTACT, u < e[None]).astype(float)
    obs[missing] = np.nan
    return obs


def tail_p(stat: np.ndarray, null: np.ndarray, min_exceed: int = 10, n_tail: int = 250) -> np.ndarray:
    """P(null >= stat): empirical when >= min_exceed null values exceed, otherwise from a
    generalized Pareto fit to the n_tail largest null values."""
    pool = np.sort(null[np.isfinite(null)])
    N = len(pool)
    exceed = N - np.searchsorted(pool, stat, side="left")
    p = (exceed + 1) / (N + 1)
    low = exceed < min_exceed
    if low.any() and N > 4 * n_tail:
        u = pool[-n_tail - 1]
        c, _, scale = st.genpareto.fit(pool[-n_tail:] - u, floc=0)
        p[low] = n_tail / N * st.genpareto.sf(stat[low] - u, c, 0, scale)
    return np.clip(p, 1e-300, 1)
