"""Empirical-Bayes configuration model over molecular layers.

Given per-layer statistics z_g (length L, N(0, R_g) under the null) the model is

    z_g ~ pi_0 N(0, R_g) + sum_{gamma != 0} sum_k pi_{gamma,k} N(0, R_g + tau_k^2 diag(gamma))

gamma in {0,1}^L marks which layers carry a convergence shift and tau_k is a
grid of effect scales. The weights pi are learned genome-wide by EM, so the
data decide which layer combinations are typical of convergence. Outputs are
the local false discovery rate lfdr_g = P(gamma = 0 | z_g), the posterior
probability that each layer is active, and the most probable configuration:
an interpretable "convergence signature" per gene.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


@dataclass
class ConfigFit:
    lfdr: np.ndarray            # (genes,)
    layer_post: np.ndarray      # (genes, layers) P(layer active | z)
    configs: np.ndarray         # (C, layers) boolean, row 0 is the null
    config_weights: np.ndarray  # (C,) learned prior mass per configuration
    map_config: np.ndarray      # (genes,) index of the most probable configuration
    loglik: float
    comp_weights: np.ndarray    # (components,) learned mixture weights
    taus: tuple
    one_sided: bool


def _mvn_logpdf(z: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Batched log N(z; 0, S) for z (G, L), S (G, L, L)."""
    Lc = np.linalg.cholesky(S)
    y = np.linalg.solve(Lc, z[..., None])[..., 0]
    logdet = 2 * np.log(np.einsum("gii->gi", Lc)).sum(-1)
    return -0.5 * ((y ** 2).sum(-1) + logdet + z.shape[1] * np.log(2 * np.pi))


def _components(L: int, taus) -> tuple[np.ndarray, list[tuple[int, float]]]:
    configs = np.array(list(itertools.product([0, 1], repeat=L)), dtype=bool)
    configs = configs[np.argsort(configs.sum(1), kind="stable")]       # null first
    comps = [(0, 0.0)] + [(c, t) for c in range(1, len(configs)) for t in taus]
    return configs, comps


def _component_logf(z: np.ndarray, R: np.ndarray, configs: np.ndarray,
                    comps: list[tuple[int, float]], one_sided: bool) -> np.ndarray:
    logf = np.empty((z.shape[0], len(comps)))
    for j, (c, t) in enumerate(comps):
        S = R + (t ** 2) * np.diag(configs[c].astype(float))[None]
        logf[:, j] = _mvn_logpdf(z, S)
        if one_sided and c > 0:
            mean_active = (z * configs[c]).sum(1) / configs[c].sum()
            logf[:, j] = np.where(mean_active > 0, logf[:, j] + np.log(2.0), -np.inf)
    return logf


def _posterior(logf: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, float]:
    a = logf + np.log(w)
    m = a.max(1, keepdims=True)
    post = np.exp(a - m)
    norm = post.sum(1, keepdims=True)
    return post / norm, float((np.log(norm[:, 0]) + m[:, 0]).sum())


def fit_configurations(z: np.ndarray, R: np.ndarray, taus=(1.0, 2.0, 3.0, 5.0),
                       n_iter: int = 200, tol: float = 1e-6, null_prior: float = 10.0,
                       one_sided: bool = True) -> ConfigFit:
    """EM for the configuration mixture.

    z: (genes, layers) per-layer z-scores; R: (genes, layers, layers) null correlation.
    null_prior: Dirichlet pseudo-count on the null component (keeps pi_0 from collapsing).
    one_sided: convergence is a positive shift; components get zero density for
    genes whose active-layer z-scores are mostly negative (the factor 2 is an
    approximate renormalisation of the truncated component).
    """
    G, L = z.shape
    configs, comps = _components(L, taus)
    logf = _component_logf(z, R, configs, comps, one_sided)
    w = np.full(len(comps), 0.1 / (len(comps) - 1))
    w[0] = 0.9
    prev = -np.inf
    for _ in range(n_iter):
        post, ll = _posterior(logf, w)
        counts = post.sum(0)
        counts[0] += null_prior
        w = np.maximum(counts / counts.sum(), 1e-12)
        if ll - prev < tol * abs(ll):
            break
        prev = ll
    comp_cfg = np.array([c for c, _ in comps])
    cfg_post = np.zeros((G, len(configs)))
    np.add.at(cfg_post.T, comp_cfg, post.T)
    cfg_w = np.zeros(len(configs))
    np.add.at(cfg_w, comp_cfg, w)
    return ConfigFit(lfdr=cfg_post[:, 0], layer_post=cfg_post @ configs.astype(float),
                     configs=configs, config_weights=cfg_w,
                     map_config=cfg_post.argmax(1), loglik=ll,
                     comp_weights=w, taus=tuple(taus), one_sided=one_sided)


def lfdr_given_fit(fit: ConfigFit, z: np.ndarray, R: np.ndarray) -> np.ndarray:
    """lfdr for new z-scores under an already-fitted prior (used on null foregrounds)."""
    configs, comps = _components(z.shape[1], fit.taus)
    post, _ = _posterior(_component_logf(z, R, configs, comps, fit.one_sided), fit.comp_weights)
    return post[:, 0]


def bayes_fdr_reject(lfdr: np.ndarray, q: float) -> np.ndarray:
    """Reject the largest set whose average lfdr is <= q."""
    order = np.argsort(lfdr)
    cum = np.cumsum(lfdr[order]) / np.arange(1, len(lfdr) + 1)
    k = int((cum <= q).sum())
    rej = np.zeros(len(lfdr), dtype=bool)
    rej[order[:k]] = True
    return rej
