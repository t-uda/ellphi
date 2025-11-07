"""Geometric helpers for ellipse and ellipsoid clouds."""

from __future__ import annotations

from typing import Tuple

import numpy

__all__ = [
    "unit_vector",
    "axes_from_cov",
    "coef_from_axes",
    "coef_from_cov",
    "pack_conic",
    "unpack_conic",
    "infer_dim_from_coef_length",
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
# Symmetric quadratic-form utilities (n-dimensional)
# ------------------------------------------------------------------

def infer_dim_from_coef_length(length: int) -> int:
    """Return the dimensionality ``n`` encoded by a coefficient vector."""

    if length < 3:
        raise ValueError("Coefficient length must encode at least a 2D conic")
    root = numpy.sqrt(8 * length + 1.0)
    n = int(round((root - 3.0) / 2.0))
    if (n + 1) * (n + 2) // 2 != length:
        raise ValueError(f"Coefficient length {length} is not valid for a conic")
    return n


def _triangular_indices(n: int) -> Tuple[numpy.ndarray, numpy.ndarray]:
    return numpy.triu_indices(n)


def pack_conic(A: numpy.ndarray, b: numpy.ndarray, c: numpy.ndarray) -> numpy.ndarray:
    """Pack ``(A, b, c)`` into flattened conic coefficients."""

    A = numpy.asarray(A, dtype=float)
    b = numpy.asarray(b, dtype=float)
    c = numpy.asarray(c, dtype=float)

    if A.ndim < 2:
        raise ValueError("A must have at least two dimensions")
    n = A.shape[-1]
    if A.shape[-2] != n:
        raise ValueError("A must be a square matrix")
    if b.shape[-1] != n:
        raise ValueError("Linear term dimensionality mismatch")

    base_shape = numpy.broadcast_shapes(A.shape[:-2], b.shape[:-1], c.shape)
    A = numpy.broadcast_to(A, base_shape + (n, n))
    b = numpy.broadcast_to(b, base_shape + (n,))
    c = numpy.broadcast_to(c, base_shape)

    tri_i, tri_j = _triangular_indices(n)
    tri_count = len(tri_i)
    total = (n + 1) * (n + 2) // 2
    out = numpy.empty(base_shape + (total,), dtype=float)

    for idx, (i, j) in enumerate(zip(tri_i, tri_j)):
        out[..., idx] = A[..., i, j]

    out[..., tri_count : tri_count + n] = b
    out[..., -1] = c
    return out


def unpack_conic(
    coef: numpy.ndarray,
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """Return ``(A, b, c)`` for the supplied conic coefficients."""

    coef = numpy.asarray(coef, dtype=float)
    squeeze = coef.ndim == 1
    if squeeze:
        coef = coef[numpy.newaxis, :]

    length = coef.shape[-1]
    n = infer_dim_from_coef_length(length)
    tri_i, tri_j = _triangular_indices(n)
    tri_count = len(tri_i)

    quad = coef[..., :tri_count]
    linear = coef[..., tri_count : tri_count + n]
    const = coef[..., -1]

    shape = coef.shape[:-1]
    A = numpy.zeros(shape + (n, n), dtype=float)
    for idx, (i, j) in enumerate(zip(tri_i, tri_j)):
        A[..., i, j] = quad[..., idx]
        if i != j:
            A[..., j, i] = quad[..., idx]

    if squeeze:
        return A[0], linear[0], numpy.asarray(const[0])
    return A, linear, const


def _inv_broadcast(cov: numpy.ndarray) -> numpy.ndarray:
    """Vectorized inverse of a batch of covariance matrices."""

    return numpy.linalg.inv(cov)


# ------------------------------------------------------------------
# Public façade
# ------------------------------------------------------------------

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
    """Centre + covariance → conic coefficients."""

    X = numpy.asarray(X, dtype=float)
    cov = numpy.asarray(cov, dtype=float)

    if X.ndim == 1:
        X = X[numpy.newaxis, :]
    if cov.ndim == 2:
        cov = cov[numpy.newaxis, :, :]

    if X.shape[0] != cov.shape[0]:
        raise ValueError("Number of centres and covariance matrices must match")
    if X.shape[1] != cov.shape[1] or cov.shape[1] != cov.shape[2]:
        raise ValueError("Covariance matrices must align with centre dimensions")

    centers = X[..., :, None]
    matrices = _inv_broadcast(cov) / scale**2
    coef_b = -(matrices @ centers)[..., 0]
    coef_c = numpy.einsum("...i,...i->...", X, -coef_b)
    return pack_conic(matrices, coef_b, coef_c)
