from __future__ import annotations
from typing import Tuple
import numpy as np
from .solver import solve_mu

__all__ = ["solve_mu_numerical_diff"]


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
