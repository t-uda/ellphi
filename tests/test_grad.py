"""Tests for ellphi.grad (tangency_grad and pdist_tangency_grad)."""

from __future__ import annotations

import numpy as np
import pytest

from ellphi import tangency_grad, pdist_tangency_grad, pdist_tangency, TangencyGrad
from ellphi.solver import tangency

from .factories import random_coef_pair, random_covariance
from ellphi.geometry import coef_from_cov


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _finite_diff_t_dp(p, q, h=1e-6):
    """Central finite difference of tangency().t w.r.t. p."""
    grad = np.zeros_like(p)
    for i in range(len(p)):
        p_plus = p.copy()
        p_plus[i] += h
        p_minus = p.copy()
        p_minus[i] -= h
        grad[i] = (tangency(p_plus, q).t - tangency(p_minus, q).t) / (2 * h)
    return grad


def _finite_diff_t_dq(p, q, h=1e-6):
    """Central finite difference of tangency().t w.r.t. q."""
    grad = np.zeros_like(q)
    for i in range(len(q)):
        q_plus = q.copy()
        q_plus[i] += h
        q_minus = q.copy()
        q_minus[i] -= h
        grad[i] = (tangency(p, q_plus).t - tangency(p, q_minus).t) / (2 * h)
    return grad


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_tangency_grad_dt_dp(rng):
    """tangency_grad().dt_dp matches central finite differences."""
    p, q = random_coef_pair(rng)
    g = tangency_grad(p, q)
    fd = _finite_diff_t_dp(p, q)
    np.testing.assert_allclose(g.dt_dp, fd, rtol=1e-5, atol=1e-8)


def test_tangency_grad_dt_dq(rng):
    """tangency_grad().dt_dq matches central finite differences."""
    p, q = random_coef_pair(rng)
    g = tangency_grad(p, q)
    fd = _finite_diff_t_dq(p, q)
    np.testing.assert_allclose(g.dt_dq, fd, rtol=1e-5, atol=1e-8)


def test_tangency_grad_returns_correct_t(rng):
    """tangency_grad().t equals tangency().t."""
    p, q = random_coef_pair(rng)
    g = tangency_grad(p, q)
    ref = tangency(p, q).t
    assert g.t == pytest.approx(ref)


def test_tangency_grad_is_frozen_dataclass(rng):
    """TangencyGrad is immutable."""
    p, q = random_coef_pair(rng)
    g = tangency_grad(p, q)
    assert isinstance(g, TangencyGrad)
    with pytest.raises((AttributeError, TypeError)):
        g.t = 999.0  # type: ignore[misc]


def test_pdist_tangency_grad_values_match(rng):
    """pdist_tangency_grad distances equal pdist_tangency to high precision."""
    n = 5
    means = rng.uniform(-20.0, 20.0, size=(n, 2))
    covs = np.stack([random_covariance(rng) for _ in range(n)])
    coefs = coef_from_cov(means, covs)

    dists, _ = pdist_tangency_grad(coefs)
    ref = pdist_tangency(coefs)

    np.testing.assert_allclose(dists, ref, atol=1e-12)


def test_pdist_tangency_grad_vjp(rng):
    """VJP directional-derivative check: g·dists_pert ≈ g·vjp(g)."""
    n = 4
    means = rng.uniform(-20.0, 20.0, size=(n, 2))
    covs = np.stack([random_covariance(rng) for _ in range(n)])
    coefs = coef_from_cov(means, covs)

    dists, vjp = pdist_tangency_grad(coefs)
    n_pairs = len(dists)

    # Random upstream gradient vector
    g_upstream = rng.standard_normal(n_pairs)
    g_coefs = vjp(g_upstream)

    # Directional derivative via finite difference
    h = 1e-5
    direction = rng.standard_normal(coefs.shape)
    direction /= np.linalg.norm(direction)
    dists_plus, _ = pdist_tangency_grad(coefs + h * direction)
    dists_minus, _ = pdist_tangency_grad(coefs - h * direction)
    fd_directional = np.dot(g_upstream, (dists_plus - dists_minus) / (2 * h))

    # VJP directional: <g_coefs, direction>
    vjp_directional = float(np.dot(g_coefs.ravel(), direction.ravel()))

    assert fd_directional == pytest.approx(vjp_directional, rel=1e-4)


def test_tangency_grad_3d(rng):
    """tangency_grad works for 3-D ellipsoids."""
    p, q = random_coef_pair(rng, dim=3)
    g = tangency_grad(p, q)
    assert g.dt_dp.shape == p.shape
    assert g.dt_dq.shape == q.shape
    fd_dp = _finite_diff_t_dp(p, q)
    fd_dq = _finite_diff_t_dq(p, q)
    np.testing.assert_allclose(g.dt_dp, fd_dp, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(g.dt_dq, fd_dq, rtol=1e-5, atol=1e-8)
