import numpy as np

from ellphi.differentiable_solver import solve_mu_numerical_diff
from ellphi.solver import solve_mu

from .factories import random_coef_pair


def solve_mu_forward_diff(
    p: np.ndarray, q: np.ndarray, *, h: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """Reference forward-difference gradients for ``solve_mu``."""
    d_mu_dp = np.zeros_like(p)
    d_mu_dq = np.zeros_like(q)
    mu_base = solve_mu(p, q)

    for i in range(len(p)):
        p_plus_h = p.copy()
        p_plus_h[i] += h
        d_mu_dp[i] = (solve_mu(p_plus_h, q) - mu_base) / h

    for i in range(len(q)):
        q_plus_h = q.copy()
        q_plus_h[i] += h
        d_mu_dq[i] = (solve_mu(p, q_plus_h) - mu_base) / h

    return d_mu_dp, d_mu_dq


def test_numerical_differentiation(rng):
    """The numerical Jacobian matches a forward-difference baseline."""
    p, q = random_coef_pair(rng)

    d_mu_dp, d_mu_dq = solve_mu_numerical_diff(p, q)
    assert d_mu_dp.shape == (6,), f"Expected shape (6,), but got {d_mu_dp.shape}"
    assert d_mu_dq.shape == (6,), f"Expected shape (6,), but got {d_mu_dq.shape}"

    d_mu_dp_forward, d_mu_dq_forward = solve_mu_forward_diff(p, q)
    np.testing.assert_allclose(
        d_mu_dp,
        d_mu_dp_forward,
        rtol=1e-4,
        atol=1e-8,
        err_msg="Central and forward difference for dp are not close enough.",
    )
    np.testing.assert_allclose(
        d_mu_dq,
        d_mu_dq_forward,
        rtol=1e-4,
        atol=1e-8,
        err_msg="Central and forward difference for dq are not close enough.",
    )
