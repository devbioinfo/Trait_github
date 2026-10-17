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
    def __init__(self, tree: Tree, log10_delta_grid: np.ndarray | None = None):
        K = shared_path_kernel(tree)
        lam, U = np.linalg.eigh(K)
        self.lam = np.clip(lam, 0.0, None)
        self.U = U
        self.grid = 10.0 ** (np.linspace(-3, 3, 31) if log10_delta_grid is None else log10_delta_grid)

    def fit(self, X: np.ndarray, fg: np.ndarray) -> LMMFit:
        G, L, B = X.shape
        Y = X.reshape(G * L, B) @ self.U                          # rotated responses
        D = self.U.T @ np.column_stack([np.ones(B), fg.astype(float)])  # rotated design (B, 2)
        p = D.shape[1]
        best_ll = np.full(G * L, -np.inf)
        out_beta = np.zeros(G * L)
        out_se = np.zeros(G * L)
        out_delta = np.zeros(G * L)
        out_s2 = np.zeros(G * L)
        for d in self.grid:
            w = 1.0 / (d * self.lam + 1.0)                       # (B,)
            DtWD = (D * w[:, None]).T @ D                         # (2, 2), shared by all genes
            inv = np.linalg.inv(DtWD)
            DtWy = (Y * w) @ D                                    # (GL, 2)
            b = DtWy @ inv                                        # (GL, 2)
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
        # whitened residuals at each gene/layer's own delta, for cross-layer correlation
        w = 1.0 / (out_delta[:, None] * self.lam[None, :] + 1.0)
        bfull = np.zeros((G * L, p))
        for d in np.unique(out_delta):
            m = out_delta == d
            wd = 1.0 / (d * self.lam + 1.0)
            DtWD = (D * wd[:, None]).T @ D
            bfull[m] = ((Y[m] * wd) @ D) @ np.linalg.inv(DtWD)
        R = (Y - bfull @ D.T) * np.sqrt(w) / np.sqrt(out_s2)[:, None]
        return LMMFit(z=z.reshape(G, L), beta=out_beta.reshape(G, L),
                      delta=out_delta.reshape(G, L), resid=R.reshape(G, L, B))


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
