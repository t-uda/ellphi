from __future__ import annotations

"""Utilities describing the tangent pencil at the solution ``μ``."""

from dataclasses import dataclass

import numpy as np


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
    """Return the 2×2 quadratic-form matrix associated with ``coef``."""

    a, b, c = coef[:3]
    return np.array([[a, b], [b, c]], dtype=float)


def linear_vector(coef: np.ndarray) -> np.ndarray:
    """Return the linear-term vector associated with ``coef``."""

    return np.array(coef[3:5], dtype=float)


def build_tangent_pencil(mu: float, p: np.ndarray, q: np.ndarray) -> TangentPencil:
    """Construct the tangent pencil for ``μ`` from ``p`` and ``q``."""

    coef = (1.0 - mu) * p + mu * q
    quad = quad_matrix(coef)
    linear = linear_vector(coef)
    det = float(quad[0, 0] * quad[1, 1] - quad[0, 1] ** 2)
    if det == 0.0:
        raise ZeroDivisionError("Degenerate conic (determinant zero)")
    inv_quad = (1.0 / det) * np.array(
        [[quad[1, 1], -quad[0, 1]], [-quad[0, 1], quad[0, 0]]],
        dtype=float,
    )
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

    jac = np.zeros((6, 2), dtype=float)
    basis_matrices = (
        np.array([[1.0, 0.0], [0.0, 0.0]], dtype=float),
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float),
        np.array([[0.0, 0.0], [0.0, 1.0]], dtype=float),
    )
    basis_vectors = (
        np.array([1.0, 0.0], dtype=float),
        np.array([0.0, 1.0], dtype=float),
    )

    for idx in range(3):
        d_quad = basis_matrices[idx]
        rhs = d_quad @ pencil.center
        jac[idx] = -(pencil.inv_quad @ rhs)

    jac[3] = -(pencil.inv_quad @ basis_vectors[0])
    jac[4] = -(pencil.inv_quad @ basis_vectors[1])
    # The constant term does not influence the center.
    return jac
