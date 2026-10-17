"""Phylogenetic linear mixed model for branch-level convergence, fitted genome-wide.

For gene g and layer l the branch values y (length B) follow

    y = b0 + beta * fg + u + e,   u ~ N(0, h2 K),   e ~ N(0, s2 I)

K[b, b'] is the length of the root-to-branch path shared by branches b and b',
i.e. the covariance of a Brownian process accumulated along the tree. With
delta = h2 / s2 the covariance is s2 (delta K + I). Diagonalising K = U diag(lam) U'
once makes every gene and layer a weighted least-squares problem with weights
1 / (delta lam + 1), so REML over a delta grid is vectorised across the whole
genome (the FaST-LMM idea). beta is tested by a Wald t statistic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as st

from .tree import Tree


def shared_path_kernel(tree: Tree) -> np.ndarray:
    A = tree.path_matrix()
    return (A * tree.length[: tree.n_branches]) @ A.T


@dataclass
class LMMFit:
    z: np.ndarray        # (genes, layers) Wald statistic for beta, mapped to N(0,1) scale
    beta: np.ndarray     # (genes, layers)
    delta: np.ndarray    # (genes, layers) REML estimate of h2 / s2
    resid: np.ndarray    # (genes, layers, branches) whitened residuals


class PhyloLMM:
    def __init__(self, tree: Tree, log10_delta_grid: np.ndarray | None = None,
                 keep: np.ndarray | None = None):
        """keep: optional boolean mask of branches to analyse (e.g. dropping the two
        root branches, which an unrooted estimate cannot tell apart). Inputs to
        `fit` must then have only the kept branches."""
        K = shared_path_kernel(tree)
        if keep is not None:
            K = K[np.ix_(keep, keep)]
        self.K = K
        lam, U = np.linalg.eigh(K)
        self.lam = np.clip(lam, 0.0, None)
        self.U = U
        self.grid = 10.0 ** (np.linspace(-3, 3, 31) if log10_delta_grid is None else log10_delta_grid)

    def fit(self, X: np.ndarray, fg: np.ndarray, branch_sd: np.ndarray | None = None) -> LMMFit:
        """branch_sd: optional (layers, branches) noise scale per branch. Estimated branch
        lengths are far noisier on short branches; with e ~ N(0, s2 diag(sd^2)) the model
        is rescaled exactly: y/sd, design/sd and K/(sd sd') (one eigendecomposition per layer).
        """
        G, L, B = X.shape
        design = np.column_stack([np.ones(B), fg.astype(float)])
        if branch_sd is None:
            z, beta, delta, resid = self._fit_block(X.reshape(G * L, B), design, self.U, self.lam)
            return LMMFit(z=z.reshape(G, L), beta=beta.reshape(G, L),
                          delta=delta.reshape(G, L), resid=resid.reshape(G, L, B))
        out = [self._fit_block(X[:, l, :] / branch_sd[l], design / branch_sd[l][:, None],
                               *self._eig_scaled(branch_sd[l])) for l in range(L)]
        z, beta, delta, resid = (np.stack([o[i] for o in out], axis=1) for i in range(4))
        return LMMFit(z=z, beta=beta, delta=delta, resid=resid)

    def _eig_scaled(self, sd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lam, U = np.linalg.eigh(self.K / np.outer(sd, sd))
        return U, np.clip(lam, 0.0, None)

    def _fit_block(self, Y0: np.ndarray, design: np.ndarray, U: np.ndarray, lam: np.ndarray):
        """REML grid fit for rows of Y0 (n, B) sharing one covariance structure."""
        n, B = Y0.shape
        Y = Y0 @ U                                                # rotated responses
        D = U.T @ design                                          # rotated design (B, 2)
        p = D.shape[1]
        best_ll = np.full(n, -np.inf)
        out_beta = np.zeros(n)
        out_se = np.zeros(n)
        out_delta = np.zeros(n)
        out_s2 = np.zeros(n)
        for d in self.grid:
            w = 1.0 / (d * lam + 1.0)                            # (B,)
            DtWD = (D * w[:, None]).T @ D                         # (2, 2), shared by all rows
            inv = np.linalg.inv(DtWD)
            DtWy = (Y * w) @ D                                    # (n, 2)
            b = DtWy @ inv                                        # (n, 2)
            rss = (Y ** 2 * w).sum(1) - (b * DtWy).sum(1)
            s2 = np.maximum(rss / (B - p), 1e-12)
            ll = -0.5 * ((B - p) * np.log(s2) - np.log(w).sum() + np.linalg.slogdet(DtWD)[1])
            better = ll > best_ll
            best_ll = np.where(better, ll, best_ll)
            out_beta = np.where(better, b[:, 1], out_beta)
            out_se = np.where(better, np.sqrt(s2 * inv[1, 1]), out_se)
            out_delta = np.where(better, d, out_delta)
            out_s2 = np.where(better, s2, out_s2)
        t = out_beta / out_se
        z = st.norm.isf(st.t.sf(t, df=B - p))                     # t -> N(0,1) scale, one-sided
        # whitened residuals at each row's own delta, for cross-layer correlation
        w = 1.0 / (out_delta[:, None] * lam[None, :] + 1.0)
        bfull = np.zeros((n, p))
        for d in np.unique(out_delta):
            m = out_delta == d
            wd = 1.0 / (d * lam + 1.0)
            DtWD = (D * wd[:, None]).T @ D
            bfull[m] = ((Y[m] * wd) @ D) @ np.linalg.inv(DtWD)
        R = (Y - bfull @ D.T) * np.sqrt(w) / np.sqrt(out_s2)[:, None]
        return z, out_beta, out_delta, R


def branch_noise_sd(X: np.ndarray, floor: float = 1e-3) -> np.ndarray:
    """Robust per-layer, per-branch spread across genes (1.4826 x MAD), shape (layers, branches),
    rescaled to geometric mean 1 per layer."""
    med = np.median(X, axis=0)
    sd = 1.4826 * np.median(np.abs(X - med), axis=0)
    sd = np.maximum(sd, floor)
    return sd / np.exp(np.log(sd).mean(1, keepdims=True))


def layer_correlation(resid: np.ndarray, shrink: float = 0.1) -> np.ndarray:
    """Per-gene cross-layer correlation of whitened residuals, shrunk toward the genome mean."""
    B = resid.shape[-1]
    Zr = resid - resid.mean(-1, keepdims=True)
    C = np.einsum("glb,gkb->glk", Zr, Zr) / (B - 1)
    sd = np.sqrt(np.einsum("gll->gl", C))
    Rg = C / sd[:, :, None] / sd[:, None, :]
    return (1 - shrink) * Rg + shrink * Rg.mean(0, keepdims=True)


def whitened_combination(z: np.ndarray, resid: np.ndarray, shrink: float = 0.1) -> np.ndarray:
    """S_g = 1' R_g^{-1} z_g / sqrt(1' R_g^{-1} 1), R_g = cross-layer residual correlation.

    Under H0, S_g ~ N(0, 1).
    """
    G, L, _ = resid.shape
    R = layer_correlation(resid, shrink)
    w = np.linalg.solve(R, np.ones((G, L, 1)))[..., 0]
    return (w * z).sum(-1) / np.sqrt(w.sum(-1))
