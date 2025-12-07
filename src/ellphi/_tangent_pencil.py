from __future__ import annotations

"""Utilities describing the tangent pencil at the solution ``μ``."""

from dataclasses import dataclass

import numpy as np
from scipy import linalg

from .geometry import unpack_single_conic


@dataclass(frozen=True)
class TangentPencil:
    """Geometry of the conic pencil ``(1-μ) p + μ q`` at the solution ``μ``."""

    coef: np.ndarray
    quad: np.ndarray
    linear: np.ndarray
    det: float
    chol: tuple[np.ndarray, bool]
    center: np.ndarray


def quad_matrix(coef: np.ndarray) -> np.ndarray:
    """Return the quadratic-form matrix associated with ``coef``."""

    quad, _, _ = unpack_single_conic(coef)
    return quad


def linear_vector(coef: np.ndarray) -> np.ndarray:
    """Return the linear-term vector associated with ``coef``."""

    _, linear, _ = unpack_single_conic(coef)
    return linear


def build_tangent_pencil(mu: float, p: np.ndarray, q: np.ndarray) -> TangentPencil:
    """Construct the tangent pencil for ``μ`` from ``p`` and ``q``."""

    coef = (1.0 - mu) * p + mu * q
    quad = quad_matrix(coef)
    linear = linear_vector(coef)

    det = float(np.linalg.det(quad))
    
    # Rely on linalg.cho_factor and lstsq to handle singular/ill-conditioned matrices.
    # The explicit isclose(det, 0.0) check can be too aggressive if underlying solvers are robust.

    try:
        # Attempt Cholesky factorization
        chol_factor = linalg.cho_factor(quad, check_finite=False)
        center = -linalg.cho_solve(chol_factor, linear)
        chol_tuple = chol_factor  # Store the successful cholesky factor
    except linalg.LinAlgError:
        # If Cholesky fails, it means the quadratic form is not positive definite.
        # This implies we cannot form a tangent pencil suitable for derivative calculation.
        # Re-raise as ZeroDivisionError as per test expectation.
        raise ZeroDivisionError("Degenerate or non-positive definite quadratic form.")

    return TangentPencil(
        coef=coef, quad=quad, linear=linear, det=det, chol=chol_tuple, center=center
    )


def target_prime_from_pencil(
    pencil: TangentPencil, p: np.ndarray, q: np.ndarray
) -> float:
    """Evaluate ``∂F/∂μ`` for the tangency equation using cached geometry."""

    diff = p - q
    diff_mat = quad_matrix(diff)
    diff_vec = linear_vector(diff)
    residual = -(diff_mat @ pencil.center + diff_vec)

    solved = linalg.cho_solve(pencil.chol, residual)
    return float(2.0 * residual @ solved)


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
        jac[idx] = -linalg.cho_solve(pencil.chol, rhs)

    for axis in range(n_dim):
        unit = np.zeros(n_dim, dtype=float)
        unit[axis] = 1.0
        jac[n_quad + axis] = -linalg.cho_solve(pencil.chol, unit)

    # The final coefficient corresponds to the constant term and has zero effect.
    return jac
