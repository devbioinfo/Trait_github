"""Simulate branch-level evidence for many genes across several molecular layers.

Each gene g contributes, for layer l and branch b,

    x[g, l, b] = h_g * P[g, l, b] + s_g * e[g, l, b] + beta * s_g * z[g, l] * fg[b]

P is a phylogenetically inherited component (Brownian increments summed along
the root-to-branch path), so a single shift deep in the tree marks a whole
clade. That clade-level structure is the confounder that makes convergence
tests anticonservative when foreground lineages are clumped. Both P and e are
correlated across layers through the matrix R, mimicking the dependence seen
between, for example, relative rates and dN/dS. z marks the layers in which a
truly convergent gene carries signal.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tree import Tree

LAYERS = (
    "relative_rate",       # RERconverge-style relative evolutionary rate
    "convergent_subst",    # CSUBST / PCOC-style convergent substitution signal
    "selection",           # aBSREL / BUSTED-style episodic selection
    "noncoding_accel",     # accelerated conserved non-coding elements near the gene
    "plm_effect",          # protein-language-model functional effect of substitutions
)

DEFAULT_LAYER_CORR = np.array([
    [1.0, 0.3, 0.5, 0.2, 0.2],
    [0.3, 1.0, 0.3, 0.1, 0.4],
    [0.5, 0.3, 1.0, 0.1, 0.3],
    [0.2, 0.1, 0.1, 1.0, 0.1],
    [0.2, 0.4, 0.3, 0.1, 1.0],
])


@dataclass
class SimulatedGenome:
    X: np.ndarray            # (genes, layers, branches)
    is_convergent: np.ndarray  # (genes,)
    active_layers: np.ndarray  # (genes, layers)


def simulate_genome(tree: Tree, fg: np.ndarray, n_genes: int, rng: np.random.Generator, *,
                    frac_convergent: float = 0.05, effect: float = 0.5,
                    p_layer_active: float = 0.6, confounding: float = 1.0,
                    noise_df: float | None = None, lineage_shift: float = 0.0,
                    layer_corr: np.ndarray = DEFAULT_LAYER_CORR) -> SimulatedGenome:
    """noise_df: Student-t branch noise (unit variance) instead of Gaussian.
    lineage_shift: genome-wide shift on the trait lineages, shared by every gene
    (e.g. faster genome-wide evolution in a trait clade); a confounder that only
    a relative (genome-normalised) transform removes.
    """
    L = layer_corr.shape[0]
    B = tree.n_branches
    chol = np.linalg.cholesky(layer_corr)
    A = tree.path_matrix()
    sd_branch = np.sqrt(tree.length[:B])

    # Gene-specific clade-effect strength (heavy-tailed) and noise scale (gene length proxy).
    h = confounding * rng.gamma(shape=0.5, scale=2.0, size=n_genes)
    s = rng.lognormal(mean=0.0, sigma=0.5, size=n_genes)

    inc = rng.standard_normal((n_genes, B, L)) @ chol.T          # layer-correlated increments
    phylo = np.einsum("ba,gal->glb", A, inc * sd_branch[None, :, None])
    if noise_df is None:
        raw = rng.standard_normal((n_genes, B, L))
    else:
        raw = rng.standard_t(noise_df, size=(n_genes, B, L)) / np.sqrt(noise_df / (noise_df - 2))
    noise = (raw @ chol.T).transpose(0, 2, 1)

    is_conv = rng.random(n_genes) < frac_convergent
    active = (rng.random((n_genes, L)) < p_layer_active) & is_conv[:, None]
    # every convergent gene carries signal in at least one layer
    empty = is_conv & ~active.any(1)
    active[empty, rng.integers(0, L, size=empty.sum())] = True

    X = (h[:, None, None] * phylo + s[:, None, None] * noise
         + effect * s[:, None, None] * active[:, :, None] * fg[None, None, :].astype(float)
         + lineage_shift * fg[None, None, :].astype(float))
    return SimulatedGenome(X=X.astype(np.float64), is_convergent=is_conv, active_layers=active)
