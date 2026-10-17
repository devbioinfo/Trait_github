"""Per-layer association statistics and multi-layer combination tests.

The proposed statistic ("whitened permulation combination", WPC) for gene g is

    S_g = 1' C_g^{-1} (T_g - m_g) / sqrt(1' C_g^{-1} 1)

where T_g is the vector of per-layer foreground-vs-background statistics and
m_g, C_g are the mean and covariance of T_g over phylogenetic permulations.
Whitening by the gene's own permulation covariance removes both the clade-level
inflation (through m_g and the diagonal of C_g) and the dependence between
layers (through the off-diagonal of C_g). Significance is the gene's empirical
permulation p-value of S_g, so no distributional assumption is made.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as st


def welch_t(X: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Welch t of foreground minus background for every gene, layer and foreground set.

    X: (genes, layers, branches); F: (sets, branches) boolean.
    Returns (genes, layers, sets).
    """
    G, L, B = X.shape
    Ff = F.astype(np.float64)
    nf = Ff.sum(1)
    nb = B - nf
    Y = X.reshape(G * L, B)
    s1 = Y @ Ff.T
    s2 = (Y ** 2) @ Ff.T
    tot1 = Y.sum(1, keepdims=True)
    tot2 = (Y ** 2).sum(1, keepdims=True)
    mf, mb = s1 / nf, (tot1 - s1) / nb
    vf = np.maximum(s2 / nf - mf ** 2, 1e-12) * nf / (nf - 1)
    vb = np.maximum((tot2 - s2) / nb - mb ** 2, 1e-12) * nb / (nb - 1)
    t = (mf - mb) / np.sqrt(vf / nf + vb / nb)
    return t.reshape(G, L, -1)


def empirical_p(obs: np.ndarray, null: np.ndarray) -> np.ndarray:
    """One-sided upper-tail permutation p-value; obs (...,), null (..., P)."""
    P = null.shape[-1]
    return (1.0 + (null >= obs[..., None]).sum(-1)) / (P + 1.0)


def fisher(p: np.ndarray) -> np.ndarray:
    """Fisher's method across the last axis (assumes independence)."""
    k = p.shape[-1]
    return st.chi2.sf(-2.0 * np.log(p).sum(-1), 2 * k)


def acat(p: np.ndarray) -> np.ndarray:
    """Cauchy combination (Liu & Xie 2020): valid under arbitrary dependence, asymptotically."""
    p = np.clip(p, 1e-15, 1 - 1e-15)
    T = np.tan((0.5 - p) * np.pi).mean(-1)
    return st.cauchy.sf(T)


def wpc(T_obs: np.ndarray, T_null: np.ndarray, ridge: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Whitened permulation combination.

    T_obs: (genes, layers); T_null: (genes, layers, perms).
    Returns (S_obs, p) with p the per-gene empirical permulation p-value.
    """
    G, L, P = T_null.shape
    m = T_null.mean(-1)
    Z = T_null - m[..., None]
    C = np.einsum("glp,gkp->glk", Z, Z) / (P - 1)
    C += ridge * np.eye(L)[None]
    w = np.linalg.solve(C, np.ones((G, L, 1)))[..., 0]      # C^{-1} 1
    norm = np.sqrt(w.sum(-1))
    S_obs = ((T_obs - m) * w).sum(-1) / norm
    S_null = np.einsum("glp,gl->gp", Z, w) / norm[:, None]
    return S_obs, empirical_p(S_obs, S_null)


def bh(p: np.ndarray, q: float) -> np.ndarray:
    """Benjamini-Hochberg rejections at level q."""
    n = len(p)
    order = np.argsort(p)
    thresh = q * np.arange(1, n + 1) / n
    passed = p[order] <= thresh
    k = np.max(np.nonzero(passed)[0]) + 1 if passed.any() else 0
    rej = np.zeros(n, dtype=bool)
    rej[order[:k]] = True
    return rej


def perm_z(T_obs: np.ndarray, T_null: np.ndarray) -> np.ndarray:
    """Standardise observed statistics by their own permulation mean and SD."""
    return (T_obs - T_null.mean(-1)) / T_null.std(-1, ddof=1)


def pooled_null_p(obs: np.ndarray, null: np.ndarray) -> np.ndarray:
    """Upper-tail p-values against a null pooled over all genes and null foregrounds."""
    ref = np.sort(null.ravel())
    return (1.0 + ref.size - np.searchsorted(ref, obs, side="left")) / (ref.size + 1.0)


def relative_transform(X: np.ndarray) -> np.ndarray:
    """Subtract, per layer and branch, the genome-wide median across genes.

    Removes lineage-wide shifts (e.g. a clade that evolves faster at every gene),
    as relative evolutionary rates do in RERconverge.
    """
    return X - np.median(X, axis=0, keepdims=True)


def run_lmm_methods(lmm, X: np.ndarray, fg: np.ndarray, null_fgs: np.ndarray,
                    branch_sd: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Model-based tests on the phylogenetic LMM z-scores.

    lmm:<layer>     per-layer Wald test
    lmm_fisher      Fisher across layers (assumes independent layers)
    lmm_acat        Cauchy combination across layers
    lmm_wc          whitened combination with a N(0,1) reference
    lmm_wc_pooled   whitened combination against the pooled control-lineage null
    ebcm            empirical-Bayes configuration model, lfdr calibrated by that null
    _layer_post     (not a p-value) EBCM posterior that each layer is active
    """
    from .configs import fit_configurations, lfdr_given_fit
    from .lmm import layer_correlation, whitened_combination
    fit = lmm.fit(X, fg, branch_sd)
    cfg = fit_configurations(fit.z, layer_correlation(fit.resid))
    out = {"_layer_post": cfg.layer_post}
    out.update({f"lmm:{i}": st.norm.sf(fit.z[:, i]) for i in range(X.shape[1])})
    p_layers = st.norm.sf(fit.z)
    out["lmm_fisher"] = fisher(p_layers)
    out["lmm_acat"] = acat(p_layers)
    S = whitened_combination(fit.z, fit.resid)
    out["lmm_wc"] = st.norm.sf(S)
    null_fits = [lmm.fit(X, nf, branch_sd) for nf in null_fgs]
    S_null = np.stack([whitened_combination(f.z, f.resid) for f in null_fits], axis=1)
    out["lmm_wc_pooled"] = pooled_null_p(S, S_null)
    # EBCM ranks genes by lfdr; calibrate against lfdr on control-lineage foregrounds
    lfdr_null = np.stack([lfdr_given_fit(cfg, f.z, layer_correlation(f.resid)) for f in null_fits], 1)
    out["ebcm"] = pooled_null_p(-cfg.lfdr, -lfdr_null)
    return out


def run_all_methods(X: np.ndarray, fg: np.ndarray, perms: np.ndarray) -> dict[str, np.ndarray]:
    """p-values from every method compared in the benchmark.

    Permulation-based p-values use a Gaussian tail after moment matching, which
    gives genome-wide resolution beyond 1/(n_perm + 1); `wpc_emp` keeps the
    purely empirical version as a check.
    """
    T_obs = welch_t(X, fg[None])[..., 0]
    T_null = welch_t(X, perms)
    p_naive = st.norm.sf(T_obs)                    # treats branches as independent
    p_perm = st.norm.sf(perm_z(T_obs, T_null))     # calibrated per layer
    out = {f"naive:{i}": p_naive[:, i] for i in range(X.shape[1])}
    out.update({f"perm:{i}": p_perm[:, i] for i in range(X.shape[1])})
    out["naive_fisher"] = fisher(p_naive)
    out["perm_fisher"] = fisher(p_perm)
    out["perm_acat"] = acat(p_perm)
    S_obs, p_emp = wpc(T_obs, T_null)
    out["wpc"] = st.norm.sf(S_obs)                 # S has unit null variance by construction
    out["wpc_emp"] = p_emp
    return out
