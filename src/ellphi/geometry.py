"""Geometric helpers for ellipsoids in arbitrary dimensions."""

from __future__ import annotations

import numpy
from numpy.typing import ArrayLike

__all__ = [
    "unit_vector",
    "axes_from_cov",
    "coef_from_axes",
    "coef_from_cov",
    "pack_conic",
    "unpack_conic",
    "infer_dim_from_coef_length",
]


def infer_dim_from_coef_length(length: int) -> int:
    """Return the spatial dimension encoded by a flattened conic."""

    if length < 6:
        raise ValueError("Coefficient array must have length at least 6 for n ≥ 2")
    disc = 1 + 8 * length
    root = int(numpy.sqrt(disc))
    if root * root != disc:
        raise ValueError("Coefficient length does not match any ellipsoid dimension")
    dim = (root - 3) // 2
    if (dim + 1) * (dim + 2) // 2 != length:
        raise ValueError("Coefficient length does not correspond to a valid ellipsoid")
    return dim


def _triangular_indices(dim: int) -> tuple[numpy.ndarray, numpy.ndarray]:
    return numpy.triu_indices(dim)


def pack_conic(
    quad: numpy.ndarray,
    linear: numpy.ndarray,
    constant: ArrayLike,
) -> numpy.ndarray:
    """Pack quadratic data ``(A, b, c)`` into a flattened coefficient vector."""

    quad = numpy.asarray(quad, dtype=float)
    linear = numpy.asarray(linear, dtype=float)
    constant = numpy.asarray(constant, dtype=float)

    if quad.ndim < 2 or quad.shape[-1] != quad.shape[-2]:
        raise ValueError("Quadratic term must be a square matrix")
    if linear.shape[-1] != quad.shape[-1]:
        raise ValueError("Linear term dimension must match quadratic term")

    dim = quad.shape[-1]
    tri_upper = _triangular_indices(dim)
    quad_flat = quad[..., tri_upper[0], tri_upper[1]]
    linear_flat = linear
    constant_flat = numpy.expand_dims(constant, axis=-1)
    return numpy.concatenate([quad_flat, linear_flat, constant_flat], axis=-1)


def unpack_conic(coef: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray, float]:
    """Return ``(A, b, c)`` from a flattened conic coefficient vector."""

    coef = numpy.asarray(coef, dtype=float)
    if coef.ndim != 1:
        raise ValueError("Expected a one-dimensional coefficient array")

    dim = infer_dim_from_coef_length(coef.shape[0])
    tri_len = dim * (dim + 1) // 2
    quad_flat = coef[:tri_len]
    linear = coef[tri_len : tri_len + dim]
    constant = float(coef[-1])

    quad = numpy.zeros((dim, dim), dtype=float)
    tri_upper = _triangular_indices(dim)
    quad[tri_upper] = quad_flat
    quad[(tri_upper[1], tri_upper[0])] = quad_flat
    return quad, linear, constant


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
    lam0, lam1 = eigvals[:, 0], eigvals[:, 1]  # ascending order: lam0 <= lam1
    v1 = eigvecs[:, 1]
    theta = numpy.arctan2(v1[:, 1], v1[:, 0])
    # Major axis, minor axis, major axis angle
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
    f = (x**2 * sin**2 - 2 * x * y * sin * cos + y**2 * cos**2) / r1**2 + (
        x**2 * cos**2 + 2 * x * y * sin * cos + y**2 * sin**2
    ) / r0**2
    return numpy.stack([a, b, c, d, e, f], axis=-1)  # (..., 6)


# ------------------------------------------------------------------
# Public façade
# ------------------------------------------------------------------


def _inv_broadcast(cov: numpy.ndarray) -> numpy.ndarray:
    """Vectorized inverse of a batch of SPD matrices."""

    return numpy.linalg.inv(cov)


def coef_from_axes(X: float, r0: float, r1: float, theta: float) -> numpy.ndarray:
    """Centre & axes → conic coefficient array (6,)."""
    return _coef_core(X, r0, r1, numpy.cos(theta), numpy.sin(theta))


def coef_from_cov(
    X: numpy.ndarray,
    cov: numpy.ndarray,
    /,
    *,
    scale: float = 1.0,
) -> numpy.ndarray:
    """Centre + covariance → conic coefficients (any dimension)."""

    X = numpy.asarray(X, dtype=float)
    cov = numpy.asarray(cov, dtype=float)

    if X.ndim == 1:
        X = X[None, :]
    if cov.ndim == 2:
        cov = cov[None, :, :]

    inv_cov = _inv_broadcast(cov) / scale**2
    centers = X[..., None]
    linear = -(inv_cov @ centers)[..., 0]
    constant = numpy.einsum("...i,...ij,...j->...", X, inv_cov, X)
    return pack_conic(inv_cov, linear, constant)
