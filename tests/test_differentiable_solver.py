import numpy as np
import pytest

from ellphi.differentiable_solver import (
    solve_mu_gradients,
    solve_mu_numerical_diff,
)
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
    assert d_mu_dp.shape == p.shape
    assert d_mu_dq.shape == q.shape

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


def test_analytic_gradients_match_central_difference(rng):
    """The analytic gradients agree with central differences."""

    for _ in range(5):
        p, q = random_coef_pair(rng)
        mu, d_mu_dp, d_mu_dq = solve_mu_gradients(p, q)
        d_mu_dp_num, d_mu_dq_num = solve_mu_numerical_diff(p, q)

        np.testing.assert_allclose(
            d_mu_dp,
            d_mu_dp_num,
            rtol=1e-6,
            atol=1e-8,
            err_msg="Analytic and numerical gradients w.r.t. p differ.",
        )
        np.testing.assert_allclose(
            d_mu_dq,
            d_mu_dq_num,
            rtol=1e-6,
            atol=1e-8,
            err_msg="Analytic and numerical gradients w.r.t. q differ.",
        )

        mu_direct = solve_mu(p, q)
        assert mu == pytest.approx(mu_direct)


def test_analytic_gradients_high_dimension(rng):
    p, q = random_coef_pair(rng, dim=3)
    mu, d_mu_dp, d_mu_dq = solve_mu_gradients(p, q)
    d_mu_dp_num, d_mu_dq_num = solve_mu_numerical_diff(p, q)

    assert d_mu_dp.shape == p.shape
    assert d_mu_dq.shape == q.shape

    np.testing.assert_allclose(d_mu_dp, d_mu_dp_num, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(d_mu_dq, d_mu_dq_num, rtol=1e-5, atol=1e-7)

    mu_direct = solve_mu(p, q)
    assert mu == pytest.approx(mu_direct)
