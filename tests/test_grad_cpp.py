"""Tests for the C++ backend of pdist_tangency_grad."""

from __future__ import annotations

import types

import numpy as np
import pytest

import ellphi._tangency_cpp as _cpp
from ellphi import pdist_tangency, pdist_tangency_grad
from ellphi.geometry import coef_from_cov
from ellphi.grad import _pdist_tangency_grad_python

from .factories import random_covariance, rotation_matrix


requires_cpp_grad = pytest.mark.skipif(
    not _cpp.has_pdist_tangency_grad(),
    reason="C++ pdist_tangency_grad kernel not available",
)


def _random_coefs(rng, n, dim=2):
    means = rng.uniform(-20.0, 20.0, size=(n, dim))
    covs = np.stack([random_covariance(rng, dim=dim) for _ in range(n)])
    return coef_from_cov(means, covs)


def _eccentric_coefs(rng, n):
    """Nearly degenerate 2-D ellipses: high aspect ratio, close centres."""
    means = rng.uniform(-1.0, 1.0, size=(n, 2))
    covs = []
    for _ in range(n):
        axes = np.array([rng.uniform(1e-3, 1e-2), rng.uniform(1.0, 2.0)])
        rot = rotation_matrix(rng.uniform(0.0, np.pi))
        covs.append(rot @ np.diag(axes) @ rot.T)
    return coef_from_cov(means, np.stack(covs))


@requires_cpp_grad
@pytest.mark.parametrize("n,dim", [(4, 2), (10, 2), (25, 2), (4, 3), (10, 3)])
def test_cpp_grad_matches_python_reference(rng, n, dim):
    """C++ kernel agrees with the pure-Python reference to <= 1e-10."""
    coefs = _random_coefs(rng, n, dim=dim)

    dists_cpp, dt_dp_cpp, dt_dq_cpp = _cpp.pdist_tangency_grad(coefs)
    dists_py, dt_dp_py, dt_dq_py = _pdist_tangency_grad_python(coefs)

    np.testing.assert_allclose(dists_cpp, dists_py, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(dt_dp_cpp, dt_dp_py, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(dt_dq_cpp, dt_dq_py, rtol=1e-10, atol=1e-10)


@requires_cpp_grad
def test_cpp_grad_matches_python_reference_eccentric(rng):
    """Agreement holds for nearly degenerate (highly eccentric) ellipses."""
    coefs = _eccentric_coefs(rng, 6)

    dists_cpp, dt_dp_cpp, dt_dq_cpp = _cpp.pdist_tangency_grad(coefs)
    dists_py, dt_dp_py, dt_dq_py = _pdist_tangency_grad_python(coefs)

    np.testing.assert_allclose(dists_cpp, dists_py, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(dt_dp_cpp, dt_dp_py, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(dt_dq_cpp, dt_dq_py, rtol=1e-10, atol=1e-10)


@requires_cpp_grad
def test_cpp_grad_dists_match_pdist_tangency(rng):
    """Distances from the grad kernel equal the forward pdist_tangency."""
    coefs = _random_coefs(rng, 8)
    dists, _ = pdist_tangency_grad(coefs)
    ref = pdist_tangency(coefs)
    np.testing.assert_allclose(dists, ref, rtol=0, atol=1e-12)


@requires_cpp_grad
def test_vjp_matches_python_path(rng, monkeypatch):
    """The public VJP gives the same pullback on both backends."""
    coefs = _random_coefs(rng, 7)
    cotangent = rng.standard_normal(7 * 6 // 2)

    dists_cpp, vjp_cpp = pdist_tangency_grad(coefs)
    grad_cpp = vjp_cpp(cotangent)

    monkeypatch.setattr(_cpp, "has_pdist_tangency_grad", lambda: False)
    dists_py, vjp_py = pdist_tangency_grad(coefs)
    grad_py = vjp_py(cotangent)

    np.testing.assert_allclose(dists_cpp, dists_py, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(grad_cpp, grad_py, rtol=1e-10, atol=1e-10)


def test_python_fallback_when_backend_missing(rng, monkeypatch):
    """Without the C++ library the Python path is selected and works."""
    coefs = _random_coefs(rng, 5)
    expected_dists, expected_dt_dp, expected_dt_dq = _pdist_tangency_grad_python(coefs)

    monkeypatch.setattr(_cpp, "_LIB", None)
    assert _cpp.is_available() is False
    assert _cpp.has_pdist_tangency_grad() is False

    dists, vjp = pdist_tangency_grad(coefs)
    np.testing.assert_allclose(dists, expected_dists, rtol=1e-10, atol=1e-10)

    cotangent = rng.standard_normal(len(dists))
    g_coefs = vjp(cotangent)
    expected = np.zeros_like(coefs)
    for k, (i, j) in enumerate((i, j) for i in range(5) for j in range(i + 1, 5)):
        expected[i] += cotangent[k] * expected_dt_dp[k]
        expected[j] += cotangent[k] * expected_dt_dq[k]
    np.testing.assert_allclose(g_coefs, expected, rtol=1e-10, atol=1e-12)


def test_stale_library_without_grad_symbol(monkeypatch):
    """A library lacking the new export reports the grad kernel as missing."""
    monkeypatch.setattr(_cpp, "_LIB", types.SimpleNamespace())
    assert _cpp.has_pdist_tangency_grad() is False


@requires_cpp_grad
def test_cpp_grad_identical_ellipsoids_raise():
    """Identical ellipsoids are degenerate and raise ZeroDivisionError."""
    means = np.array([[0.3, -0.2], [0.3, -0.2]])
    covs = np.stack([np.eye(2), np.eye(2)])
    coefs = coef_from_cov(means, covs)
    with pytest.raises(ZeroDivisionError):
        _cpp.pdist_tangency_grad(coefs)


@requires_cpp_grad
def test_cpp_grad_concentric_ellipsoids_raise():
    """Concentric nested ellipsoids raise ZeroDivisionError as in Python."""
    means = np.array([[0.5, 0.5], [0.5, 0.5]])
    covs = np.stack([np.eye(2), 4.0 * np.eye(2)])
    coefs = coef_from_cov(means, covs)
    with pytest.raises(ZeroDivisionError):
        _cpp.pdist_tangency_grad(coefs)


def test_single_ellipsoid_empty_result(rng):
    """N=1 produces empty distances and a zero VJP on any backend."""
    coefs = _random_coefs(rng, 1)
    dists, vjp = pdist_tangency_grad(coefs)
    assert dists.shape == (0,)
    g_coefs = vjp(np.zeros(0))
    np.testing.assert_array_equal(g_coefs, np.zeros_like(coefs))
