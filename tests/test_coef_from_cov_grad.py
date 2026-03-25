"""Tests for ellphi.grad.coef_from_cov_grad."""

from __future__ import annotations

import numpy as np
import pytest

from ellphi import coef_from_cov_grad
from ellphi.geometry import coef_from_cov

from .factories import random_covariance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _numerical_jacobian_X(centers, covs, scale, h=1e-6):
    """Central FD Jacobian of coef_from_cov w.r.t. centers."""
    n, d = centers.shape
    coefs0 = coef_from_cov(centers, covs, scale=scale)
    m = coefs0.shape[-1]
    jac = np.zeros((n, m, n, d))
    for i in range(n):
        for j in range(d):
            X_plus = centers.copy()
            X_plus[i, j] += h
            X_minus = centers.copy()
            X_minus[i, j] -= h
            c_plus = coef_from_cov(X_plus, covs, scale=scale)
            c_minus = coef_from_cov(X_minus, covs, scale=scale)
            jac[:, :, i, j] = (c_plus - c_minus) / (2 * h)
    return jac


def _numerical_jacobian_cov(centers, covs, scale, h=1e-6):
    """Central FD Jacobian of coef_from_cov w.r.t. cov entries.

    Perturbs each entry of the symmetric covariance matrix while maintaining
    symmetry. For the symmetric parameterization, we perturb both (j,k) and
    (k,j) together and split the off-diagonal derivative evenly across the
    returned matrix entries.
    """
    n, d = centers.shape
    coefs0 = coef_from_cov(centers, covs, scale=scale)
    m = coefs0.shape[-1]
    jac = np.zeros((n, m, n, d, d))
    for i in range(n):
        for j in range(d):
            for k in range(j, d):
                cov_plus = covs.copy()
                cov_plus[i, j, k] += h
                if j != k:
                    cov_plus[i, k, j] += h
                cov_minus = covs.copy()
                cov_minus[i, j, k] -= h
                if j != k:
                    cov_minus[i, k, j] -= h
                c_plus = coef_from_cov(centers, cov_plus, scale=scale)
                c_minus = coef_from_cov(centers, cov_minus, scale=scale)
                deriv = (c_plus - c_minus) / (2 * h)
                if j == k:
                    jac[:, :, i, j, k] = deriv
                else:
                    jac[:, :, i, j, k] = 0.5 * deriv
                    jac[:, :, i, k, j] = 0.5 * deriv
    return jac


def _check_vjp_against_fd(centers, covs, scale, rng):
    """Check VJP matches numerical Jacobian for given inputs."""
    n, d = centers.shape

    coefs, vjp = coef_from_cov_grad(centers, covs, scale=scale)
    m = coefs.shape[-1]

    # Check forward pass matches coef_from_cov
    ref = coef_from_cov(centers, covs, scale=scale)
    np.testing.assert_allclose(coefs, ref, atol=1e-12)

    # Random upstream gradient
    g_coefs = rng.standard_normal((n, m))
    grad_X, grad_cov = vjp(g_coefs)

    # --- Check grad_X via FD ---
    jac_X = _numerical_jacobian_X(centers, covs, scale)
    # Expected grad_X[i, j] = sum over (a, b) of g_coefs[a, b] * jac_X[a, b, i, j]
    expected_grad_X = np.einsum("ab,abij->ij", g_coefs, jac_X)
    np.testing.assert_allclose(grad_X, expected_grad_X, rtol=1e-5, atol=1e-8)

    # --- Check grad_cov via FD ---
    np.testing.assert_allclose(grad_cov, np.swapaxes(grad_cov, -2, -1), atol=1e-12)
    jac_cov = _numerical_jacobian_cov(centers, covs, scale)
    expected_grad_cov = np.einsum("ab,abijk->ijk", g_coefs, jac_cov)
    np.testing.assert_allclose(grad_cov, expected_grad_cov, rtol=1e-5, atol=1e-8)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_forward_matches_coef_from_cov_d2(rng):
    """Forward pass matches coef_from_cov for d=2."""
    centers = rng.uniform(-5, 5, size=(3, 2))
    covs = np.stack([random_covariance(rng, dim=2) for _ in range(3)])
    coefs, _ = coef_from_cov_grad(centers, covs)
    ref = coef_from_cov(centers, covs)
    np.testing.assert_allclose(coefs, ref, atol=1e-12)


def test_vjp_d2_identity_cov(rng):
    """VJP correct for d=2 with identity covariance."""
    centers = rng.uniform(-5, 5, size=(2, 2))
    covs = np.stack([np.eye(2)] * 2)
    _check_vjp_against_fd(centers, covs, scale=1.0, rng=rng)


def test_vjp_d2_random_cov(rng):
    """VJP correct for d=2 with random covariances."""
    centers = rng.uniform(-5, 5, size=(3, 2))
    covs = np.stack([random_covariance(rng, dim=2) for _ in range(3)])
    _check_vjp_against_fd(centers, covs, scale=1.0, rng=rng)


def test_vjp_d3(rng):
    """VJP correct for d=3."""
    centers = rng.uniform(-5, 5, size=(2, 3))
    covs = np.stack([random_covariance(rng, dim=3) for _ in range(2)])
    _check_vjp_against_fd(centers, covs, scale=1.0, rng=rng)


def test_vjp_rotated_cov(rng):
    """VJP correct for non-identity covariance with rotation (d=2)."""
    angle = rng.uniform(0, np.pi)
    cos, sin = np.cos(angle), np.sin(angle)
    rot = np.array([[cos, -sin], [sin, cos]])
    diag = np.diag([3.0, 0.5])
    cov = rot @ diag @ rot.T

    centers = rng.uniform(-5, 5, size=(2, 2))
    covs = np.stack([cov, cov])
    _check_vjp_against_fd(centers, covs, scale=1.0, rng=rng)


def test_vjp_with_scale(rng):
    """VJP correct for non-unit scale values."""
    centers = rng.uniform(-5, 5, size=(2, 2))
    covs = np.stack([random_covariance(rng, dim=2) for _ in range(2)])
    for scale in [0.5, 2.0, 3.7]:
        _check_vjp_against_fd(centers, covs, scale=scale, rng=rng)


def test_vjp_single_ellipsoid(rng):
    """VJP works for a single ellipsoid (d,) input."""
    center = rng.uniform(-5, 5, size=(2,))
    cov = random_covariance(rng, dim=2)

    coefs, vjp = coef_from_cov_grad(center, cov)
    ref = coef_from_cov(center, cov)
    np.testing.assert_allclose(coefs, ref, atol=1e-12)

    m = coefs.shape[-1]
    g = rng.standard_normal((1, m))
    grad_X, grad_cov = vjp(g)
    assert grad_X.shape == (1, 2)
    assert grad_cov.shape == (1, 2, 2)


def test_vjp_d4(rng):
    """VJP correct for d=4 (higher dimension)."""
    centers = rng.uniform(-3, 3, size=(2, 4))
    covs = np.stack([random_covariance(rng, dim=4) for _ in range(2)])
    _check_vjp_against_fd(centers, covs, scale=1.0, rng=rng)


def test_grad_shapes(rng):
    """Output shapes are correct."""
    n, d = 4, 3
    centers = rng.uniform(-5, 5, size=(n, d))
    covs = np.stack([random_covariance(rng, dim=d) for _ in range(n)])

    coefs, vjp = coef_from_cov_grad(centers, covs)
    m = (d + 1) * (d + 2) // 2
    assert coefs.shape == (n, m)

    g = rng.standard_normal((n, m))
    grad_X, grad_cov = vjp(g)
    assert grad_X.shape == (n, d)
    assert grad_cov.shape == (n, d, d)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_batch_size_mismatch():
    """Mismatched centre/covariance batch sizes raise ValueError."""
    with pytest.raises(ValueError, match="Mismatch"):
        coef_from_cov_grad(np.zeros((3, 2)), np.eye(2)[np.newaxis])


def test_non_square_cov():
    """Non-square covariance raises ValueError."""
    with pytest.raises(ValueError, match="square"):
        coef_from_cov_grad(np.zeros((2, 2)), np.zeros((2, 3, 2)))


def test_dimension_mismatch():
    """Centre/covariance dimension mismatch raises ValueError."""
    with pytest.raises(ValueError, match="dimensionality"):
        coef_from_cov_grad(np.zeros((2, 2)), np.eye(3)[np.newaxis].repeat(2, axis=0))


def test_singular_cov_returns_nan():
    """Singular covariance returns NaN coefficients, matching coef_from_cov."""
    centers = np.zeros((1, 2))
    cov_sing = np.zeros((1, 2, 2))

    coefs, vjp = coef_from_cov_grad(centers, cov_sing)
    ref = coef_from_cov(centers, cov_sing)
    np.testing.assert_array_equal(np.isnan(coefs), np.isnan(ref))

    grad_X, grad_cov = vjp(np.ones_like(coefs))
    assert np.all(np.isnan(grad_X))
    assert np.all(np.isnan(grad_cov))
