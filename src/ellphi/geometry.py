"""
ellphi.geometry  –  geometric helpers for ellipse cloud
=======================================================

Key API (all return NumPy float64):

- unit_vector(theta)
- axes_from_cov(cov, scale=1.0)
- coef_from_axes(x, y, r0, r1, theta) # centre+axes → (6,)
- coef_from_array(arr)                # (N,5) → (N,6)
- coef_from_cov(x, y, cov, scale=1.0) # centre+cov → (6,)
"""

from __future__ import annotations

from collections import namedtuple

import numpy
from scipy.spatial.distance import pdist, squareform 

__all__ = [
    "unit_vector",
    "axes_from_cov",
    "coef_from_axes",
    "coef_from_array",
    "coef_from_cov",
    "ellipse_cloud",
    "EllipseCloudResult"
]

# ------------------------------------------------------------------
# Pure helpers
# ------------------------------------------------------------------
def unit_vector(theta: float) -> numpy.ndarray:  # noqa: D401
    """Return the unit vector (cosθ, sinθ)."""
    return numpy.array([numpy.cos(theta), numpy.sin(theta)], dtype=float)

def axes_from_cov(cov: numpy.ndarray, /, *, scale: float = 1.0):
    """Covariance (2×2) → (r0, r1, θ) with r0 ≥ r1."""
    if len(cov.shape) <= 2:
        cov = cov[None, :, :]
    eigvals, eigvecs = numpy.linalg.eigh(cov)
    lam0, lam1 = eigvals[:, 0], eigvals[:, 1] # ascending order: lam0 <= lam1
    v1 = eigvecs[:, 1]
    theta = numpy.arctan2(v1[:, 1], v1[:, 0])
    ## Major axis, minor axis, major axis angle
    return (numpy.sqrt(lam1) * scale, numpy.sqrt(lam0) * scale, theta)

# ------------------------------------------------------------------
# Shared core formula (broadcast-friendly)
# ------------------------------------------------------------------
def _coef_core(x, y, r0, r1, cos, sin):
    """Return stacked [a,b,c,d,e,f] along last dimension."""
    a = sin**2 / r1**2 + cos**2 / r0**2
    b = (-sin * cos) / r1**2 + (sin * cos) / r0**2
    c = cos**2 / r1**2 + sin**2 / r0**2
    d = (-x * sin**2 + y * sin * cos) / r1**2 - (x * cos**2 + y * sin * cos) / r0**2
    e = (x * sin * cos - y * cos**2) / r1**2 - (x * sin * cos + y * sin**2) / r0**2
    f = (
        (x**2 * sin**2 - 2 * x * y * sin * cos + y**2 * cos**2) / r1**2
        + (x**2 * cos**2 + 2 * x * y * sin * cos + y**2 * sin**2) / r0**2
    )
    return numpy.stack([a, b, c, d, e, f], axis=-1)  # (..., 6)

# ------------------------------------------------------------------
# Public façade
# ------------------------------------------------------------------
def coef_from_axes(
        x: float,
        y: float,
        r0: float,
        r1: float,
        theta: float
) -> numpy.ndarray:
    """Centre & axes → conic coefficient array (6,)."""
    return _coef_core(x, y, r0, r1, *unit_vector(theta))

def coef_from_array(params: numpy.ndarray) -> numpy.ndarray:
    """Vectorised `(N,5)` → `(N,6)` conversion."""
    x, y, r0, r1, th = params.T
    return _coef_core(x, y, r0, r1, numpy.cos(th), numpy.sin(th))

def coef_from_cov_composed(
    x: float,
    y: float,
    cov: numpy.ndarray,
    /,
    *,
    scale: float = 1.0,
) -> numpy.ndarray:
    """Centre + covariance → conic coefficients."""
    return coef_from_axes(x, y, *axes_from_cov(cov, scale=scale))

def coef_from_cov(
    x: numpy.ndarray,
    y: numpy.ndarray,
    cov: numpy.ndarray,
    /,
    *,
    scale: float = 1.0,
) -> numpy.ndarray:
    """Centre + covariance → conic coefficients."""
    if len(cov.shape) <= 2:
        cov = cov[None, :, :]
    centers = numpy.transpose([x, y])[:, :, None]
    matrices = numpy.linalg.inv(cov) / scale**2
    coef_b = - matrices @ centers
    coef_c = centers.transpose(0, 2, 1) @ matrices @ centers
    return numpy.stack([
        matrices[:, 0, 0],
        matrices[:, 0, 1],
        matrices[:, 1, 1],
        coef_b[:, 0].ravel(),
        coef_b[:, 1].ravel(),
        coef_c.ravel()
    ], axis=-1)

EllipseCloudResult = namedtuple("EllipseCloudResult", [
    "coefs", "means", "covs", "neighbors", "k", "num_ellipses"
])

def ellipse_cloud(x: numpy.ndarray, y: numpy.ndarray, k: int, *, scale: float = 1.0) -> EllipseCloudResult:
    points = numpy.transpose([x, y])
    d = squareform(pdist(points)) # Euclidean distance matrix
    # argsort したものから :near だけとると重複が生じるので削る
    near_subsets = numpy.unique(numpy.argsort(d, axis=1)[:, :k], axis=0)
    # 各サブセットをソートしてタプルに変換
    sorted_subsets = [tuple(sorted(subset)) for subset in near_subsets]
    unique_subsets = numpy.unique(sorted_subsets, axis=0) # 重複を取り除く
    num_ellipses = unique_subsets.shape[0]
    knbd = points[unique_subsets]
    means = numpy.mean(knbd, axis=1)
    rel_nbd = knbd - means[:, None, :]
    covs = rel_nbd.transpose(0, 2, 1) @ rel_nbd / (k - 1)
    coefs = coef_from_cov(*means.T, covs)
    return EllipseCloudResult(coefs, means, covs, unique_subsets, k, num_ellipses)

