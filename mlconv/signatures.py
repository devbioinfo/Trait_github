"""Convergence signatures with data-driven effect covariances (mash-style).

For gene g with per-layer z-scores z_g and null correlation R_g, the true effect vector
mu_g has prior

    mu_g ~ pi_0 delta_0 + sum_k sum_w pi_{k,w} N(0, w^2 U_k)

The U_k are (i) canonical patterns (each subset of layers shifting independently) and
(ii) data-driven rank-one patterns u u' learned by clustering the directions of the
strongest genes, so that layers which shift together in reality (e.g. protein rate and
the selection proxy under a change in omega) form one component. w runs over a grid wide
enough for z-scores of 20 or more. Mixture weights are fitted by EM (Urbut et al. 2019,
Nat Genet 51:187). Outputs per gene and layer: posterior mean effect and local false
sign rate (lfsr); per gene: local FDR (posterior mass of the null).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy import stats as st


@dataclass
class SignatureFit:
    lfdr: np.ndarray          # (G,)
    post_mean: np.ndarray     # (G, L)
    lfsr: np.ndarray          # (G, L)
    patterns: list            # list of (name, U) prior covariance patterns
    pattern_weights: np.ndarray  # (K,) summed over scales
    top_pattern: np.ndarray   # (G,) index of the most probable pattern (-1 = null)


def data_driven_patterns(z: np.ndarray, n_patterns: int = 4, min_strength: float = 4.0,
                         n_iter: int = 50, seed: int = 0) -> list[np.ndarray]:
    """Unit directions from spherical k-means on genes whose largest |z| >= min_strength."""
    strong = z[np.abs(z).max(1) >= min_strength]
    if len(strong) < n_patterns:
        return []
    X = strong / np.linalg.norm(strong, axis=1, keepdims=True)
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), n_patterns, replace=False)]
    for _ in range(n_iter):
        lab = (X @ C.T).argmax(1)
        for k in range(n_patterns):
            if (lab == k).any():
                c = X[lab == k].mean(0)
                C[k] = c / np.linalg.norm(c)
    return [c for c in C]


def _logpdf(z: np.ndarray, S: np.ndarray) -> np.ndarray:
    Lc = np.linalg.cholesky(S)
    y = np.linalg.solve(Lc, z[..., None])[..., 0]
    return -0.5 * ((y ** 2).sum(-1) + 2 * np.log(np.einsum("gii->gi", Lc)).sum(-1)
                   + z.shape[1] * np.log(2 * np.pi))


def fit_signatures(z: np.ndarray, R: np.ndarray, layer_names: list[str], n_patterns: int = 4,
                   scales=(1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0), null_prior: float = 10.0,
                   n_iter: int = 300, tol: float = 1e-7) -> SignatureFit:
    G, L = z.shape
    patterns: list[tuple[str, np.ndarray]] = []
    for cfg in itertools.product([0, 1], repeat=L):
        if any(cfg):
            name = "+".join(n for n, c in zip(layer_names, cfg) if c)
            patterns.append((name, np.diag(np.array(cfg, dtype=float))))
    for i, u in enumerate(data_driven_patterns(z, n_patterns)):
        name = "learned:" + ",".join(f"{n}={v:+.2f}" for n, v in zip(layer_names, u))
        patterns.append((name, np.outer(u, u)))
    comps = [(k, w) for k in range(len(patterns)) for w in scales]
    logf = np.empty((G, 1 + len(comps)))
    logf[:, 0] = _logpdf(z, R)
    for j, (k, w) in enumerate(comps, start=1):
        logf[:, j] = _logpdf(z, R + (w ** 2) * patterns[k][1][None])
    pi = np.full(1 + len(comps), 0.1 / len(comps))
    pi[0] = 0.9
    prev = -np.inf
    for _ in range(n_iter):
        a = logf + np.log(pi)
        m = a.max(1, keepdims=True)
        post = np.exp(a - m)
        norm = post.sum(1, keepdims=True)
        post /= norm
        ll = float((np.log(norm[:, 0]) + m[:, 0]).sum())
        counts = post.sum(0)
        counts[0] += null_prior
        pi = np.maximum(counts / counts.sum(), 1e-12)
        if ll - prev < tol * abs(ll):
            break
        prev = ll
    # posterior moments of mu per component: m = U (U + R)^-1 z, V = U - U (U + R)^-1 U
    mean = np.zeros((G, L))
    p_pos = np.zeros((G, L))
    p_zero = post[:, :1].repeat(L, 1).copy()
    for j, (k, w) in enumerate(comps, start=1):
        U = (w ** 2) * patterns[k][1]
        A = np.linalg.solve(R + U[None], np.broadcast_to(U, (G, L, L)))    # (U+R)^-1 U, per gene
        mk = np.einsum("gkl,gk->gl", A, z)                                  # U (U+R)^-1 z
        Vk = U[None] - np.einsum("lk,gkm->glm", U, A)
        sd = np.sqrt(np.clip(np.einsum("gll->gl", Vk), 0, None))
        degenerate = sd < 1e-9                                              # layer not in pattern
        pp = np.where(degenerate, 0.0, st.norm.sf(0, mk, np.where(degenerate, 1, sd)))
        mean += post[:, j:j + 1] * mk
        p_pos += post[:, j:j + 1] * pp
        p_zero += post[:, j:j + 1] * degenerate
    p_neg = 1 - p_pos - p_zero
    lfsr = np.minimum(p_pos + p_zero, p_neg + p_zero)
    comp_k = np.array([-1] + [k for k, _ in comps])
    pat_post = np.zeros((G, len(patterns) + 1))
    np.add.at(pat_post.T, comp_k + 1, post.T)
    pw = np.zeros(len(patterns))
    np.add.at(pw, comp_k[1:], pi[1:])
    return SignatureFit(lfdr=post[:, 0], post_mean=mean, lfsr=lfsr, patterns=patterns,
                        pattern_weights=pw, top_pattern=pat_post.argmax(1) - 1)
