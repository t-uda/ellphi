"""Geometry helpers shared across the :mod:`ellphi` package.

This module historically provided 2D-only utilities for working with
ellipses represented as quadratic forms.  For the n-dimensional extension
the public façade is kept backwards compatible while the core routines now
work with generic symmetric positive-definite matrices.

Key API (all return ``numpy.float64`` arrays):

* :func:`unit_vector`
* :func:`axes_from_cov`
* :func:`coef_from_axes`
* :func:`coef_from_cov`
* :func:`pack_conic`
* :func:`unpack_conic`
* :func:`unpack_single_conic`
* :func:`infer_dim_from_coef_length`
"""

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
    "unpack_single_conic",
    "infer_dim_from_coef_length",
]


# ------------------------------------------------------------------
# Pure helpers
# ------------------------------------------------------------------
def unit_vector(theta: float) -> numpy.ndarray:
    """Computes the unit vector for a given angle.

    Args:
        theta: The angle in radians.

    Returns:
        The unit vector (cosθ, sinθ) as a NumPy array.
    """
    return numpy.transpose([numpy.cos(theta), numpy.sin(theta)])


def axes_from_cov(cov: numpy.ndarray, /, *, scale: float = 1.0):
    """Computes the axes and orientation of an ellipse from a covariance matrix.

    This function takes a 2x2 covariance matrix and returns the major axis,
    minor axis, and angle of the major axis. The axes are scaled by the
    provided scale factor.

    Args:
        cov: A 2x2 covariance matrix.
        scale: A scaling factor for the axes.

    Returns:
        A tuple containing the major axis, minor axis, and angle of the major
        axis in radians. The major axis is always greater than or equal to the
        minor axis.
    """
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


def infer_dim_from_coef_length(length: int) -> int:
    """Infers the ambient dimension `n` from the flattened coefficient length.

    The coefficient vector stores the upper triangular part of the quadratic
    matrix, followed by the linear term and the constant term. Its length is
    therefore `m = (n + 1)(n + 2) / 2`. The function validates that the
    supplied length matches this formula and returns `n`.

    Args:
        length: The length of the flattened coefficient vector.

    Returns:
        The inferred ambient dimension `n`.

    Raises:
        ValueError: If the length does not correspond to a valid dimension.
    """
    if length < 6:
        raise ValueError("Coefficient vector too short to represent a conic")
    disc = 1 + 8 * length
    sqrt_disc = int(round(numpy.sqrt(disc)))
    if sqrt_disc * sqrt_disc != disc:
        raise ValueError(
            f"Coefficient length {length} does not correspond to a symmetric "
            "quadratic form"
        )
    n = (sqrt_disc - 3) // 2
    if n < 2 or ((n + 1) * (n + 2)) // 2 != length:
        raise ValueError(
            f"Coefficient length {length} does not correspond to a valid " "dimension"
        )
    return int(n)


def _broadcast_shapes(
    quad_shape: Tuple[int, ...],
    linear_shape: Tuple[int, ...],
    const_shape: Tuple[int, ...],
) -> Tuple[int, ...]:
    """Return the shared broadcast shape for quadratic, linear and constant terms."""

    return numpy.broadcast_shapes(quad_shape, linear_shape, const_shape)


def pack_conic(
    matrices: numpy.ndarray, linear: numpy.ndarray, constant: numpy.ndarray
) -> numpy.ndarray:
    """Packs a conic's components into a flattened coefficient vector.

    This function takes the quadratic, linear, and constant terms of a conic's
    equation and packs them into a single, flattened NumPy array.

    Args:
        matrices: A NumPy array of symmetric positive-definite matrices `A` with
            shape `(..., n, n)`.
        linear: A NumPy array of linear coefficients `b` with shape `(..., n)`.
        constant: A NumPy array of the constant term `c` with shape `(...)`.

    Returns:
        A NumPy array of shape `(..., m)` where `m = (n + 1)(n + 2) / 2`,
        representing the packed conic coefficients.
    """
    matrices = numpy.asarray(matrices, dtype=float)
    linear = numpy.asarray(linear, dtype=float)
    constant = numpy.asarray(constant, dtype=float)

    if matrices.ndim < 2 or matrices.shape[-1] != matrices.shape[-2]:
        raise ValueError("Quadratic matrices must have shape (..., n, n)")
    n = matrices.shape[-1]
    if linear.shape[-1] != n:
        raise ValueError("Linear term incompatible with quadratic matrix dimension")

    quad_shape = matrices.shape[:-2]
    linear_shape = linear.shape[:-1]
    const_shape = constant.shape
    broadcast_shape = _broadcast_shapes(quad_shape, linear_shape, const_shape)

    matrices = numpy.broadcast_to(matrices, broadcast_shape + (n, n))
    linear = numpy.broadcast_to(linear, broadcast_shape + (n,))
    constant = numpy.broadcast_to(constant, broadcast_shape)

    tri_i, tri_j = numpy.triu_indices(n)
    quad_entries = matrices[..., tri_i, tri_j]
    packed = numpy.concatenate(
        [quad_entries, linear, constant[..., numpy.newaxis]], axis=-1
    )
    return packed


def unpack_conic(
    coef: numpy.ndarray,
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """Unpacks a flattened coefficient vector into a conic's components.

    This function is the inverse of :func:`pack_conic`. It takes a flattened
    coefficient vector and returns the quadratic, linear, and constant terms
    of the conic's equation.

    Args:
        coef: A NumPy array of packed conic coefficients.

    Returns:
        A tuple containing the quadratic matrix `A`, the linear coefficients
        `b`, and the constant term `c`.
    """
    coef = numpy.asarray(coef, dtype=float)
    squeeze = False
    if coef.ndim == 1:
        coef = coef[numpy.newaxis, :]
        squeeze = True
    if coef.ndim != 2:
        raise ValueError("Coefficient array must be one- or two-dimensional")

    length = coef.shape[-1]
    n = infer_dim_from_coef_length(length)
    tri_i, tri_j = numpy.triu_indices(n)
    n_quad = tri_i.size

    quad_entries = coef[:, :n_quad]
    linear = coef[:, n_quad : n_quad + n]
    constant = coef[:, n_quad + n]

    matrices = numpy.zeros((coef.shape[0], n, n), dtype=float)
    matrices[:, tri_i, tri_j] = quad_entries
    matrices[:, tri_j, tri_i] = quad_entries

    if squeeze:
        return matrices[0], linear[0], constant[0]
    return matrices, linear, constant


def unpack_single_conic(
    coef: numpy.ndarray,
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """Unpacks a single flattened coefficient vector into a conic's components.

    This function is a convenience wrapper around :func:`unpack_conic` for a
    single conic. It accepts a coefficient array of shape `(m,)` or `(1, m)`
    and returns the conic's components.

    Args:
        coef: A NumPy array of packed conic coefficients for a single conic.

    Returns:
        A tuple containing the quadratic matrix `A`, the linear coefficients
        `b`, and the constant term `c`.
    """
    matrices, linear, constant = unpack_conic(coef)
    if matrices.ndim == 3:
        if matrices.shape[0] != 1:
            raise ValueError(
                "Expected coefficients for a single conic, "
                f"received {matrices.shape[0]} entries"
            )
        matrices = matrices[0]
        linear = linear[0]
        constant = constant[0]
    return matrices, linear, constant


def coef_from_axes(X: float, r0: float, r1: float, theta: float) -> numpy.ndarray:
    """Computes the conic coefficient array from the center and axes.

    Args:
        X: The center of the conic.
        r0: The major axis.
        r1: The minor axis.
        theta: The angle of the major axis in radians.

    Returns:
        A NumPy array of shape (6,) representing the conic coefficients.
    """
    return _coef_core(X, r0, r1, numpy.cos(theta), numpy.sin(theta))


def coef_from_cov(
    X: numpy.ndarray,
    cov: numpy.ndarray,
    /,
    *,
    scale: float = 1.0,
) -> numpy.ndarray:
    """Converts centers and covariances to packed conic coefficients.

    This function takes an array of centers and covariance matrices and
    converts them into packed conic coefficients.

    Args:
        X: An array of centers with shape `(n, d)` or a single center with
            shape `(d,)`. Single centers are promoted to `(1, d)`.
        cov: An array of covariance matrices with shape `(n, d, d)` or a
            single covariance with shape `(d, d)`.
        scale: An optional scaling factor for the covariance matrices.

    Returns:
        A NumPy array of packed conic coefficients with shape `(n, m)`, where
        `m = (d + 1)(d + 2) / 2`.
    """
    centers = numpy.asarray(X, dtype=float)
    cov = numpy.asarray(cov, dtype=float)

    if centers.ndim == 1:
        centers = centers[numpy.newaxis, :]
    if cov.ndim == 2:
        cov = cov[numpy.newaxis, :, :]

    if centers.shape[0] != cov.shape[0]:
        raise ValueError("Mismatch between number of centres and covariance matrices")
    if cov.shape[-1] != cov.shape[-2]:
        raise ValueError("Covariance matrices must be square")

    n_dim = centers.shape[-1]
    if cov.shape[-1] != n_dim:
        raise ValueError("Centre dimensionality and covariance size must agree")

    try:
        inv_cov = numpy.linalg.inv(cov)
    except numpy.linalg.LinAlgError:
        # If the covariance matrix is singular,
        # we cannot compute the conic coefficients.
        # We return an array of NaNs to indicate that the computation failed,
        # allowing the caller (e.g., the solver) to handle this case gracefully.
        m = (n_dim + 1) * (n_dim + 2) // 2
        return numpy.full(centers.shape[:-1] + (m,), numpy.nan, dtype=float)

    matrices = inv_cov / (scale**2)
    b = -(matrices @ centers[..., None])[..., 0]
    c = numpy.einsum("...i,...ij,...j->...", centers, matrices, centers)
    packed = pack_conic(matrices, b, c)
    return packed
