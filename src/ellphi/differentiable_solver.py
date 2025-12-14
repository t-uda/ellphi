from __future__ import annotations
from typing import Tuple

import numpy as np
from ._tangent_pencil import (
    TangentPencil,
    build_tangent_pencil,
    center_jacobian,
    linear_vector,
    quad_matrix,
    target_prime_from_pencil,
)
from .solver import MethodName, solve_mu

__all__ = ["solve_mu_gradients", "solve_mu_numerical_diff"]


def solve_mu_numerical_diff(
    p: np.ndarray, q: np.ndarray, h: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes `solve_mu` partial derivatives using central differences.

    This function calculates the partial derivatives of `solve_mu` with
    respect to `p` and `q` using the central difference method for
    numerical differentiation.

    Args:
        p: The coefficient vector of the first ellipse.
        q: The coefficient vector of the second ellipse.
        h: The step size for the finite difference calculation.

    Returns:
        A tuple containing the gradients `(d_mu_dp, d_mu_dq)`.
    """
    d_mu_dp = np.zeros_like(p)
    d_mu_dq = np.zeros_like(q)

    for i in range(len(p)):
        p_plus_h = p.copy()
        p_plus_h[i] += h
        p_minus_h = p.copy()
        p_minus_h[i] -= h
        d_mu_dp[i] = (solve_mu(p_plus_h, q) - solve_mu(p_minus_h, q)) / (2 * h)

    for i in range(len(q)):
        q_plus_h = q.copy()
        q_plus_h[i] += h
        q_minus_h = q.copy()
        q_minus_h[i] -= h
        d_mu_dq[i] = (solve_mu(p, q_plus_h) - solve_mu(p, q_minus_h)) / (2 * h)

    return d_mu_dp, d_mu_dq


def solve_mu_gradients(
    p: np.ndarray,
    q: np.ndarray,
    *,
    mu: float | None = None,
    method: MethodName = "brentq+newton",
    bracket: Tuple[float, float] = (0.0, 1.0),
    x0: float | None = None,
    hybrid_bracket_maxiter: int | None = None,
    hybrid_newton_maxiter: int | None = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Returns `μ` and its partial derivatives with respect to `p` and `q`.

    Args:
        p: The coefficient vector of the first conic.
        q: The coefficient vector of the second conic.
        mu: An optional pre-computed value of `μ`. If not provided, it will be
            computed using `solve_mu`.
        method: The root-finding method to use if `mu` is not provided.
        bracket: The bracketing interval to use if `mu` is not provided.
        x0: An optional initial guess for Newton's method if `mu` is not
            provided.
        hybrid_bracket_maxiter: An optional maximum number of iterations for
            the bracket phase in the hybrid method.
        hybrid_newton_maxiter: An optional maximum number of iterations for
            the Newton phase in the hybrid method.

    Returns:
        A tuple containing the solved `μ`, and its partial derivatives
        `∂μ/∂p` and `∂μ/∂q`.
    """
    if mu is None:
        mu = solve_mu(
            p,
            q,
            method=method,
            bracket=bracket,
            x0=x0,
            hybrid_bracket_maxiter=hybrid_bracket_maxiter,
            hybrid_newton_maxiter=hybrid_newton_maxiter,
        )

    pencil: TangentPencil = build_tangent_pencil(mu, p, q)
    diff = p - q
    diff_mat = quad_matrix(diff)
    diff_vec = linear_vector(diff)
    residual = -(diff_mat @ pencil.center + diff_vec)

    dF_dmu = target_prime_from_pencil(pencil, p, q)
    if np.isclose(dF_dmu, 0.0):
        raise ZeroDivisionError("Derivative with respect to mu is numerically zero")

    phi_x = -2.0 * residual
    center = pencil.center
    n_dim = center.shape[0]
    tri_i, tri_j = np.triu_indices(n_dim)
    quad_entries = np.empty(tri_i.size, dtype=float)
    for idx, (i, j) in enumerate(zip(tri_i, tri_j)):
        if i == j:
            quad_entries[idx] = center[i] ** 2
        else:
            quad_entries[idx] = 2.0 * center[i] * center[j]
    linear_entries = 2.0 * center
    base = np.concatenate([quad_entries, linear_entries, np.array([1.0])])

    jac_center = center_jacobian(pencil)
    dx_dp = (1.0 - mu) * jac_center
    dx_dq = mu * jac_center

    chain_dp = dx_dp @ phi_x
    chain_dq = dx_dq @ phi_x

    dF_dp = base + chain_dp
    dF_dq = -base + chain_dq

    d_mu_dp = -dF_dp / dF_dmu
    d_mu_dq = -dF_dq / dF_dmu

    return mu, d_mu_dp, d_mu_dq
