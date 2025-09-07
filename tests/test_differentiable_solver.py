import numpy as np
from ellphi.solver import solve_mu
from ellphi.differentiable_solver import solve_mu_numerical_diff
from ellphi.geometry import coef_from_cov


def generate_ellipse_pair(seed=42):
    """Generate a pair of ellipses for testing."""
    np.random.seed(seed)
    # Generate two distinct means
    means = np.random.rand(2, 2) * 100
    covs_list = []
    for _ in range(2):
        a = np.random.rand() * 5 + 1
        b = np.random.rand() * 5 + 1
        angle = np.random.rand() * np.pi
        rot = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        cov = rot @ np.diag([a, b]) @ rot.T
        covs_list.append(cov)
    covs = np.array(covs_list)
    p = coef_from_cov(means[0], covs[0])[0]
    q = coef_from_cov(means[1], covs[1])[0]

    return p, q


def solve_mu_forward_diff(p, q, h=1e-6):
    """A forward difference implementation for comparison."""
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


def test_numerical_differentiation():
    """
    Test the numerical differentiation of solve_mu.
    """
    p, q = generate_ellipse_pair()
    # Test shape of the output
    d_mu_dp, d_mu_dq = solve_mu_numerical_diff(p, q)
    assert d_mu_dp.shape == (6,), f"Expected shape (6,), but got {d_mu_dp.shape}"
    assert d_mu_dq.shape == (6,), f"Expected shape (6,), but got {d_mu_dq.shape}"

    # Test precision by comparing with forward difference
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
