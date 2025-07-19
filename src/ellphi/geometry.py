"""
ellphi.geometry  –  geometric helpers for ellipse cloud
=======================================================

Key API (all return NumPy float64):

- unit_vector(theta)
- axes_from_cov(cov, scale=1.0)
- coef_from_axes(x, y, r1, r2, theta) # centre+axes → (6,)
- coef_from_array(arr)                # (N,5) → (N,6)
- coef_from_cov(x, y, cov, scale=1.0) # centre+cov → (6,)
"""

from __future__ import annotations

from collections import namedtuple

import numpy

__all__ = [
    "unit_vector",
    "axes_from_cov",
    "coef_from_axes",
    "coef_from_array",
    "coef_from_cov",
]

# ------------------------------------------------------------------
# Pure helpers
# ------------------------------------------------------------------
def unit_vector(theta: float) -> numpy.ndarray:  # noqa: D401
    """Return the unit vector (cosθ, sinθ)."""
    return numpy.array([numpy.cos(theta), numpy.sin(theta)], dtype=float)

def axes_from_cov(cov: numpy.ndarray, /, *, scale: float = 1.0):
    """Covariance (2×2) → (r1, r2, θ) with r1 ≥ r2."""
    eigvals, eigvecs = numpy.linalg.eigh(cov)
    order = numpy.argsort(eigvals)[::-1]
    lam1, lam2 = eigvals[order]
    v1 = eigvecs[:, order[0]]
    return float(numpy.sqrt(lam1) * scale), float(numpy.sqrt(lam2) * scale), float(
        numpy.arctan2(v1[1], v1[0])
    )

# ------------------------------------------------------------------
# Shared core formula (broadcast-friendly)
# ------------------------------------------------------------------
def _coef_core(x, y, r1, r2, cos, sin):
    """Return stacked [a,b,c,d,e,f] along last dimension."""
    a = sin**2 / r2**2 + cos**2 / r1**2
    b = (-sin * cos) / r2**2 + (sin * cos) / r1**2
    c = cos**2 / r2**2 + sin**2 / r1**2
    d = (-x * sin**2 + y * sin * cos) / r2**2 - (x * cos**2 + y * sin * cos) / r1**2
    e = (x * sin * cos - y * cos**2) / r2**2 - (x * sin * cos + y * sin**2) / r1**2
    f = (
        (x**2 * sin**2 - 2 * x * y * sin * cos + y**2 * cos**2) / r2**2
        + (x**2 * cos**2 + 2 * x * y * sin * cos + y**2 * sin**2) / r1**2
    )
    return numpy.stack([a, b, c, d, e, f], axis=-1)  # (..., 6)

# ------------------------------------------------------------------
# Public façade
# ------------------------------------------------------------------
def coef_from_axes(
        x: float,
        y: float,
        r1: float,
        r2: float,
        theta: float
) -> numpy.ndarray:
    """Centre & axes → conic coefficient array (6,)."""
    return _coef_core(x, y, r1, r2, *unit_vector(theta)).ravel()

def coef_from_array(params: numpy.ndarray) -> numpy.ndarray:
    """Vectorised `(N,5)` → `(N,6)` conversion."""
    x, y, r1, r2, th = params.T
    return _coef_core(x, y, r1, r2, numpy.cos(th), numpy.sin(th))

def coef_from_cov(
    x: float,
    y: float,
    cov: numpy.ndarray,
    /,
    *,
    scale: float = 1.0,
) -> numpy.ndarray:
    """Centre + covariance → conic coefficients."""
    return coef_from_axes(x, y, *axes_from_cov(cov, scale=scale))

