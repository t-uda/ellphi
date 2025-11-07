from __future__ import annotations

"""Utilities describing the tangent pencil at the solution ``μ``."""

from dataclasses import dataclass

import numpy as np

from .geometry import unpack_conic


@dataclass(frozen=True)
class TangentPencil:
    """Geometry of the conic pencil ``(1-μ) p + μ q``."""

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
    quad, linear, _ = unpack_conic(coef)
    try:
        inv_quad = np.linalg.inv(quad)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - defensive
        raise ZeroDivisionError("Degenerate conic (determinant zero)") from exc
    det = float(np.linalg.det(quad))
    if det == 0.0:
        raise ZeroDivisionError("Degenerate conic (determinant zero)")
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


def _quadratic_basis(dim: int) -> np.ndarray:
    basis = []
    for i in range(dim):
        for j in range(i, dim):
            mat = np.zeros((dim, dim), dtype=float)
            mat[i, j] = 1.0
            mat[j, i] = 1.0
            basis.append(mat)
    return np.asarray(basis)


def center_jacobian(pencil: TangentPencil) -> np.ndarray:
    """Return ``∂x_c/∂r`` where ``r`` are pencil coefficients."""

    dim = pencil.center.shape[0]
    quad_basis = _quadratic_basis(dim)
    num_quad = quad_basis.shape[0]
    num_coef = num_quad + dim + 1
    jac = np.zeros((num_coef, dim), dtype=float)

    inv_quad = pencil.inv_quad
    center = pencil.center

    for idx, basis in enumerate(quad_basis):
        rhs = basis @ center
        jac[idx] = -(inv_quad @ rhs)

    for idx in range(dim):
        basis_vec = np.zeros(dim, dtype=float)
        basis_vec[idx] = 1.0
        jac[num_quad + idx] = -(inv_quad @ basis_vec)

    # Constant term leaves the center unchanged.
    return jac
