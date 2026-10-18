"""Generalised least squares with a genome-learned null covariance.

Every gene evolves on the same tree, so the B x B covariance of branch values
across genes can be estimated from the genome itself. That single matrix absorbs
phylogenetic structure, clade-level rate jumps, branch-specific estimation noise
and correlated estimation errors between neighbouring branches, none of which a
parametric kernel captures exactly when branch lengths are estimated from
sequences. Each gene keeps its own scale, so the model is

    y_g = b0 + beta * f + e_g,   e_g ~ N(0, s_g^2 Sigma)

and beta is tested by a Wald t (df = B - 2) after whitening with Sigma^{-1/2}.
Requires many more genes than branches (G >> B); real genomes have G ~ 10^4.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as st


def learn_covariance(Y: np.ndarray, exclude: np.ndarray | None = None, ridge: float = 1e-6) -> np.ndarray:
    """Covariance of per-gene standardised branch values. Y: (genes, branches).

    exclude: genes left out of the estimate (e.g. strong candidates, so that true
    convergence does not inflate variance along the foreground direction).
    """
    Z = Y if exclude is None else Y[~exclude]
    Z = Z - Z.mean(1, keepdims=True)
    Z = Z / Z.std(1, keepdims=True)
    S = np.cov(Z, rowvar=False)
    return S + ridge * np.trace(S) / S.shape[0] * np.eye(S.shape[0])


def gls_z(Y: np.ndarray, Sigma: np.ndarray, fgs: np.ndarray) -> np.ndarray:
    """One-sided z-scores for the foreground effect. Y (G, B); fgs (M, B) -> (G, M)."""
    G, B = Y.shape
    Li = np.linalg.inv(np.linalg.cholesky(Sigma))
    Yw = Y @ Li.T
    one = Li @ np.ones(B)
    out = np.empty((G, len(fgs)))
    for m, f in enumerate(np.atleast_2d(fgs)):
        Dw = np.column_stack([one, Li @ f.astype(float)])
        inv = np.linalg.inv(Dw.T @ Dw)
        b = Yw @ (Dw @ inv)
        res = Yw - b @ Dw.T
        s2 = (res ** 2).sum(1) / (B - 2)
        t = b[:, 1] / np.sqrt(s2 * inv[1, 1])
        out[:, m] = st.norm.isf(st.t.sf(t, B - 2))
    return out


def gls_residuals(Y: np.ndarray, Sigma: np.ndarray, fg: np.ndarray) -> np.ndarray:
    """Whitened, scaled residuals (G, B) under the foreground model, for cross-layer correlation."""
    G, B = Y.shape
    Li = np.linalg.inv(np.linalg.cholesky(Sigma))
    Yw = Y @ Li.T
    Dw = np.column_stack([Li @ np.ones(B), Li @ fg.astype(float)])
    b = Yw @ (Dw @ np.linalg.inv(Dw.T @ Dw))
    res = Yw - b @ Dw.T
    return res / np.sqrt((res ** 2).sum(1, keepdims=True) / (B - 2))


def trimmed_covariance(Y: np.ndarray, fg: np.ndarray, trim: float = 0.1, n_iter: int = 2) -> np.ndarray:
    """Learn Sigma, drop the genes with the largest foreground z, and re-learn."""
    Sigma = learn_covariance(Y)
    for _ in range(n_iter):
        z = gls_z(Y, Sigma, fg[None])[:, 0]
        Sigma = learn_covariance(Y, exclude=z > np.quantile(z, 1 - trim))
    return Sigma


def cross_origin_excess(Sigma: np.ndarray, origin: np.ndarray, K: np.ndarray, n_bins: int = 20) -> float:
    """Excess covariance between foreground branches of different independent origins.

    origin: (B,) origin index for foreground branches, -1 for background.
    Under the null, the covariance of two branches depends on their shared ancestry
    K[b, b']; it is predicted for each cross-origin pair from background pairs with
    similar K (binned medians). Convergent genes add the same amount kappa to every
    foreground pair, so the mean excess over cross-origin pairs estimates kappa.
    """
    B = len(origin)
    iu = np.triu_indices(B, 1)
    s, k = Sigma[iu], K[iu]
    o1, o2 = origin[iu[0]], origin[iu[1]]
    bg = (o1 < 0) & (o2 < 0)
    cross = (o1 >= 0) & (o2 >= 0) & (o1 != o2)
    if not cross.any():
        return 0.0
    edges = np.unique(np.quantile(k[bg], np.linspace(0, 1, n_bins + 1)))
    idx_bg = np.clip(np.searchsorted(edges, k[bg], side="right") - 1, 0, len(edges) - 2)
    med = np.array([np.median(s[bg][idx_bg == j]) if (idx_bg == j).any() else 0.0
                    for j in range(len(edges) - 1)])
    idx_c = np.clip(np.searchsorted(edges, k[cross], side="right") - 1, 0, len(edges) - 2)
    return float(np.mean(s[cross] - med[idx_c]))


def decontaminated_covariance(Y: np.ndarray, origin: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, float]:
    """Genome-learned covariance with the rank-one convergence term removed.

    Sigma_all ~ Sigma_null + kappa f f'; kappa is estimated by `cross_origin_excess`
    (clipped at 0) and subtracted. Returns (Sigma_null, kappa).
    """
    S = learn_covariance(Y)
    kappa = max(cross_origin_excess(S, origin, K), 0.0)
    f = (origin >= 0).astype(float)
    S0 = S - kappa * np.outer(f, f)
    w, V = np.linalg.eigh(S0)
    S0 = (V * np.maximum(w, 1e-6 * w.max())) @ V.T          # keep positive definite
    return S0, kappa


def gls_origin_z(Y: np.ndarray, Sigma: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Per-origin one-sided z-scores from a GLS fit with one indicator per origin.

    origin: (B,) origin index for foreground branches, -1 for background.
    Returns (G, n_origins). Each origin's effect is estimated jointly with the others,
    against the same background.
    """
    G, B = Y.shape
    m = int(origin.max()) + 1
    Li = np.linalg.inv(np.linalg.cholesky(Sigma))
    Yw = Y @ Li.T
    X = np.column_stack([np.ones(B)] + [(origin == i).astype(float) for i in range(m)])
    Xw = Li @ X
    inv = np.linalg.inv(Xw.T @ Xw)
    b = Yw @ (Xw @ inv)
    res = Yw - b @ Xw.T
    df = B - X.shape[1]
    s2 = (res ** 2).sum(1) / df
    t = b[:, 1:] / np.sqrt(s2[:, None] * np.diag(inv)[1:][None])
    return st.norm.isf(st.t.sf(t, df))


def partial_conjunction_p(p: np.ndarray, r: int, method: str = "fisher") -> np.ndarray:
    """Benjamini & Heller (2008) partial conjunction p-value for H0: fewer than r of the
    m tests are non-null. Uses the m - r + 1 largest p-values. p: (G, m)."""
    m = p.shape[1]
    tail = np.sort(p, axis=1)[:, r - 1:]                 # p_(r), ..., p_(m)
    if method == "fisher":
        return st.chi2.sf(-2 * np.log(tail).sum(1), 2 * (m - r + 1))
    if method == "bonferroni":
        return np.minimum(1.0, (m - r + 1) * tail[:, 0])
    raise ValueError(method)


def leave_one_origin_out_p(Y: np.ndarray, Sigma: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """max over origins i of the pooled GLS p-value computed with origin i's branches removed.

    An intersection-union test: a gene is significant only if its foreground signal
    survives the removal of any single independent origin, so a rate shift confined
    to one lineage cannot by itself produce a convergence call. Removed branches are
    dropped from the data and from Sigma (not reassigned to the background).
    """
    m = int(origin.max()) + 1
    p = np.zeros((Y.shape[0], m))
    for i in range(m):
        keep = origin != i
        f = (origin[keep] >= 0)
        if not f.any():                                   # nothing left to test
            p[:, i] = 1.0
            continue
        z = gls_z(Y[:, keep], Sigma[np.ix_(keep, keep)], f[None])[:, 0]
        p[:, i] = st.norm.sf(z)
    return p.max(1)


# ---------------------------------------------------------------------------
# Missing branches (genes in which some species have no ortholog)
# ---------------------------------------------------------------------------

def learn_covariance_missing(Y: np.ndarray, min_pairs: int = 50) -> np.ndarray:
    """Pairwise-complete covariance of per-gene standardised branch values; NaN = missing.

    Each gene is centred and scaled on its observed branches; covariances use all genes
    observed at both branches (at least `min_pairs`, else the pair is set to 0). The result
    is projected to the nearest positive-definite matrix by eigenvalue clipping.
    """
    Z = Y - np.nanmean(Y, 1, keepdims=True)
    Z = Z / np.nanstd(Z, 1, keepdims=True)
    M = (~np.isnan(Z)).astype(float)
    Z0 = np.nan_to_num(Z)
    n = M.T @ M
    S = (Z0.T @ Z0) / np.maximum(n - 1, 1)
    S[n < min_pairs] = 0.0
    S = (S + S.T) / 2
    w, V = np.linalg.eigh(S)
    w = np.maximum(w, 1e-4 * w.max())
    return (V * w) @ V.T


def gls_z_missing(Y: np.ndarray, Sigma: np.ndarray, f: np.ndarray, min_branches: int = 20,
                  min_fg: int = 2) -> np.ndarray:
    """One-sided GLS z-scores per gene using only its observed branches (NaN = missing).

    Genes sharing a missing-data pattern are fitted together. Genes with fewer than
    `min_branches` observed branches or `min_fg` observed foreground branches get NaN.
    """
    G, B = Y.shape
    obs = ~np.isnan(Y)
    z = np.full(G, np.nan)
    patterns, inverse = np.unique(obs, axis=0, return_inverse=True)
    for k, pat in enumerate(patterns):
        rows = np.nonzero(inverse.ravel() == k)[0]
        if pat.sum() < min_branches or (f[pat] > 0).sum() < min_fg:
            continue
        z[rows] = gls_z(Y[np.ix_(rows, pat)], Sigma[np.ix_(pat, pat)], f[pat][None])[:, 0]
    return z


def leave_one_origin_out_p_missing(Y: np.ndarray, Sigma: np.ndarray, origin: np.ndarray,
                                   **kw) -> np.ndarray:
    """Leave-one-origin-out intersection-union test allowing missing branches."""
    m = int(origin.max()) + 1
    p = np.ones((Y.shape[0], m))
    for i in range(m):
        keep = origin != i
        f = (origin[keep] >= 0).astype(float)
        z = gls_z_missing(Y[:, keep], Sigma[np.ix_(keep, keep)], f, **kw)
        p[:, i] = np.where(np.isnan(z), 1.0, st.norm.sf(np.nan_to_num(z)))
    return p.max(1)
