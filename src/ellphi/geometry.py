"""
ellphi.geometry  –  geometric helpers for ellipse cloud
=======================================================

Key API (all return NumPy float64):

- unit_vector(theta)
- axes_from_cov(cov, scale=1.0)
- coef_from_axes(X, r0, r1, theta) # centre+axes → (6,)
- coef_from_cov(X, cov, scale=1.0) # centre+cov → (6,)
"""

from __future__ import annotations

from collections import namedtuple

import numpy

__all__ = [
    "unit_vector",
    "axes_from_cov",
    "coef_from_axes",
    "coef_from_cov",
]

# ------------------------------------------------------------------
# Pure helpers
# ------------------------------------------------------------------
def unit_vector(theta: float) -> numpy.ndarray:  # noqa: D401
    """Return the unit vector (cosθ, sinθ)."""
    return numpy.transpose([numpy.cos(theta), numpy.sin(theta)])

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
def _coef_core(X, r0, r1, cos, sin):
    """Return stacked [a,b,c,d,e,f] along last dimension."""
    x, y = numpy.transpose(X)
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
        X: float,
        r0: float,
        r1: float,
        theta: float
) -> numpy.ndarray:
    """Centre & axes → conic coefficient array (6,)."""
    return _coef_core(X, r0, r1, numpy.cos(theta), numpy.sin(theta))

def coef_from_cov_composed(
    X: float,
    cov: numpy.ndarray,
    /,
    *,
    scale: float = 1.0,
) -> numpy.ndarray:
    """Centre + covariance → conic coefficients."""
    return coef_from_axes(X, *axes_from_cov(cov, scale=scale))

def coef_from_cov(
    X: numpy.ndarray,
    cov: numpy.ndarray,
    /,
    *,
    scale: float = 1.0,
) -> numpy.ndarray:
    """Centre + covariance → conic coefficients."""
    X = numpy.array(X)
    if len(X.shape) <= 1:
        X = X[None, :] # Extend if single observation
    if len(cov.shape) <= 2:
        cov = cov[None, :, :] # Extend if single observation
    centers = X[:, :, None]
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

