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
    """
    Computes the partial derivatives of solve_mu with respect to p and q
    using the central difference method.

    Parameters
    ----------
    p : np.ndarray
        Coefficient vector of the first ellipse.
    q : np.ndarray
        Coefficient vector of the second ellipse.
    h : float, optional
        The step size for the finite difference calculation, by default 1e-6.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the gradients (d_mu_dp, d_mu_dq).
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
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Return ``μ`` and its partial derivatives with respect to ``p`` and ``q``.

    Parameters
    ----------
    p, q:
        Coefficient vectors defining the two conics.
    mu:
        Optional pre-computed value of ``μ``. When omitted the function
        solves for ``μ`` using :func:`ellphi.solver.solve_mu` with the
        supplied keyword arguments.
    method, bracket, x0:
        Parameters forwarded to :func:`ellphi.solver.solve_mu` when ``mu``
        is not given.

    Returns
    -------
    Tuple[float, np.ndarray, np.ndarray]
        The solved ``μ`` together with ``∂μ/∂p`` and ``∂μ/∂q``.
    """

    if mu is None:
        mu = solve_mu(p, q, method=method, bracket=bracket, x0=x0)

    pencil: TangentPencil = build_tangent_pencil(mu, p, q)
    diff = p - q
    diff_mat = quad_matrix(diff)
    diff_vec = linear_vector(diff)
    residual = -(diff_mat @ pencil.center + diff_vec)

    dF_dmu = target_prime_from_pencil(pencil, p, q)
    if np.isclose(dF_dmu, 0.0):
        raise ZeroDivisionError("Derivative with respect to mu is numerically zero")

    phi_x = -2.0 * residual
    xc0, xc1 = pencil.center
    base = np.array([xc0**2, 2.0 * xc0 * xc1, xc1**2, 2.0 * xc0, 2.0 * xc1, 1.0])

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
