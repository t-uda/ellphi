from __future__ import annotations

"""Utilities describing the tangent pencil at the solution ``μ``."""

from dataclasses import dataclass

import numpy as np

from .geometry import unpack_conic


@dataclass(frozen=True)
class TangentPencil:
    """Geometry of the conic pencil ``(1-μ) p + μ q`` at the solution ``μ``."""

    coef: np.ndarray
    quad: np.ndarray
    linear: np.ndarray
    det: float
    inv_quad: np.ndarray
    center: np.ndarray


def quad_matrix(coef: np.ndarray) -> np.ndarray:
    """Return the quadratic-form matrix associated with ``coef``."""

    quad, _, _ = unpack_conic(coef)
    return quad


def linear_vector(coef: np.ndarray) -> np.ndarray:
    """Return the linear-term vector associated with ``coef``."""

    _, linear, _ = unpack_conic(coef)
    return linear


def build_tangent_pencil(mu: float, p: np.ndarray, q: np.ndarray) -> TangentPencil:
    """Construct the tangent pencil for ``μ`` from ``p`` and ``q``."""

    coef = (1.0 - mu) * p + mu * q
    quad = quad_matrix(coef)
    linear = linear_vector(coef)
    det = float(np.linalg.det(quad))
    if np.isclose(det, 0.0):
        raise ZeroDivisionError("Degenerate conic (determinant zero)")
    inv_quad = np.linalg.inv(quad)
    center = -inv_quad @ linear
    return TangentPencil(
        coef=coef, quad=quad, linear=linear, det=det, inv_quad=inv_quad, center=center
    )


def target_prime_from_pencil(
    pencil: TangentPencil, p: np.ndarray, q: np.ndarray
) -> float:
    """Evaluate ``∂F/∂μ`` for the tangency equation using cached geometry."""

    diff = p - q
    diff_mat = quad_matrix(diff)
    diff_vec = linear_vector(diff)
    residual = -(diff_mat @ pencil.center + diff_vec)
    return float(2.0 * residual @ pencil.inv_quad @ residual)


def center_jacobian(pencil: TangentPencil) -> np.ndarray:
    """Return ``∂x_c/∂r`` where ``r`` are pencil coefficients."""

    n_dim = pencil.center.shape[0]
    tri_i, tri_j = np.triu_indices(n_dim)
    n_quad = tri_i.size
    jac = np.zeros(((n_dim + 1) * (n_dim + 2) // 2, n_dim), dtype=float)

    for idx, (i, j) in enumerate(zip(tri_i, tri_j)):
        basis = np.zeros_like(pencil.quad)
        basis[i, j] = 1.0
        if i != j:
            basis[j, i] = 1.0
        rhs = basis @ pencil.center
        jac[idx] = -(pencil.inv_quad @ rhs)

    for axis in range(n_dim):
        unit = np.zeros(n_dim, dtype=float)
        unit[axis] = 1.0
        jac[n_quad + axis] = -(pencil.inv_quad @ unit)

    # The final coefficient corresponds to the constant term and has zero effect.
    return jac
