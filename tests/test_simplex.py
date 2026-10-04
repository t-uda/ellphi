"""Tests for the pairwise-aligned many-body tangency API."""

from __future__ import annotations

from typing import get_args

import numpy as np
import pytest

import ellphi
from ellphi._minimax_python import MethodName
from ellphi.geometry import coef_from_cov

from .factories import random_coef_pair, random_covariance


def _random_coefs(k: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-2.0, 2.0, size=(k, d))
    covariances = np.stack([random_covariance(rng, dim=d) for _ in range(k)])
    return coef_from_cov(centers, covariances)


def _relative_error(actual: np.ndarray | float, expected: np.ndarray | float) -> float:
    numerator = float(np.linalg.norm(actual - expected))
    denominator = max(
        float(np.linalg.norm(actual)), float(np.linalg.norm(expected)), 1e-15
    )
    return numerator / denominator


def test_pairwise_agrees_with_tangency(solver_backend, rng):
    p, q = random_coef_pair(rng, dim=3)
    result = ellphi.tangency_simplex(np.stack((p, q)))
    pairwise = ellphi.tangency(p, q, backend=solver_backend)

    assert result.t == pytest.approx(pairwise.t, rel=1e-9)
    np.testing.assert_allclose(result.point, pairwise.point, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(
        result.mu, (1.0 - pairwise.mu, pairwise.mu), rtol=1e-8, atol=1e-10
    )


def test_translated_pair_uses_engine_value_and_centred_constraints(solver_backend):
    centers = np.array([[1e8, 0.0], [1e8 + 2.0, 0.0]])
    coefs = coef_from_cov(centers, np.repeat(np.eye(2)[None], 2, axis=0))
    pairwise_coefs = coef_from_cov(
        np.array([[0.0, 0.0], [2.0, 0.0]]), np.repeat(np.eye(2)[None], 2, axis=0)
    )

    result = ellphi.tangency_simplex(coefs)
    pairwise = ellphi.tangency(
        pairwise_coefs[0], pairwise_coefs[1], backend=solver_backend
    )
    gradient = ellphi.tangency_simplex_grad(coefs)
    tri_i, tri_j = np.triu_indices(2)
    basis = np.concatenate(
        (
            np.where(
                tri_i == tri_j,
                result.point[tri_i] ** 2,
                2.0 * result.point[tri_i] * result.point[tri_j],
            ),
            2.0 * result.point,
            np.array([1.0]),
        )
    )
    pairwise_identity = 0.5 * basis / (2.0 * pairwise.t)

    assert result.t == pytest.approx(pairwise.t, rel=1e-9)
    assert result.active_set == (0, 1)
    assert np.all(np.isfinite(gradient.dt_dcoef))
    assert _relative_error(gradient.dt_dcoef[0], pairwise_identity) < 1e-6
    assert _relative_error(gradient.dt_dcoef[1], pairwise_identity) < 1e-6


@pytest.mark.parametrize("dimension", [2, 3, 4])
@pytest.mark.parametrize("condition_number", [1e2, 1e6, 1e10, 1e12])
def test_coef_from_cov_accepts_rotated_spd_conditioning(dimension, condition_number):
    rng = np.random.default_rng(2700 + dimension)
    rotation, _ = np.linalg.qr(rng.standard_normal((dimension, dimension)))
    eigenvalues = np.geomspace(1.0, condition_number, dimension)
    covariance = rotation @ np.diag(eigenvalues) @ rotation.T
    center = rng.uniform(-2.0, 2.0, size=(1, dimension))
    coefs = coef_from_cov(center, covariance[None])

    result = ellphi.tangency_simplex(coefs)

    assert result.t == 0.0
    assert np.all(np.isfinite(result.point))


def test_scaled_centres_from_cov_are_accepted():
    centers = np.array([[1e6, -1e6], [1e6 + 2.0, -1e6 + 1.0]])
    covariances = np.array(
        [
            [[2.0, 0.3], [0.3, 1.5]],
            [[1.2, -0.2], [-0.2, 0.8]],
        ]
    )

    result = ellphi.tangency_simplex(coef_from_cov(centers, covariances))

    assert np.isfinite(result.t)
    assert result.active_set == (0, 1)


def test_packed_one_dimensional_input_is_rejected_like_pairwise():
    coefs = np.array([[1.0, -2.0, 1.0], [1.0, -4.0, 4.0]])

    with pytest.raises(
        ValueError, match="packed input requires d >= 2, as for ellphi.tangency"
    ):
        ellphi.tangency_simplex(coefs)


@pytest.mark.parametrize("method", list(get_args(MethodName)))
def test_right_triangle_distinguishes_support_and_active_set(method):
    centers = np.array([[2.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
    coefs = coef_from_cov(centers, np.repeat(np.eye(2)[None], 3, axis=0))

    result = ellphi.tangency_simplex(coefs, method=method, active_tol=1e-12)

    assert result.t == pytest.approx(np.sqrt(2.0), rel=1e-12)
    np.testing.assert_allclose(result.point, [1.0, 1.0], atol=1e-12)
    assert len(result.support) == 2
    assert len(result.active_set) == 3
    assert result.support == (0, 1)
    assert set(result.support) <= set(result.active_set)
    assert result.active_set == (0, 1, 2)


def test_singleton_and_empty_simplex():
    coefs = coef_from_cov(np.array([[1.0, -2.0]]), np.eye(2)[None])
    result = ellphi.tangency_simplex(coefs)

    assert result.t == 0.0
    np.testing.assert_allclose(result.point, [1.0, -2.0])
    np.testing.assert_array_equal(result.mu, [1.0])
    assert result.support == (0,)
    assert result.active_set == (0,)

    with pytest.raises(ValueError, match="at least one vertex"):
        ellphi.tangency_simplex(np.empty((0, coefs.shape[1])))


def test_non_convergence_raises_with_diagnostics():
    coefs = _random_coefs(5, 3, seed=101)
    with pytest.raises(RuntimeError) as exc_info:
        ellphi.tangency_simplex(coefs, method="fw+bisect", max_iter=1, tol=1e-15)

    message = str(exc_info.value)
    assert "method='fw+bisect'" in message
    assert "n_iter=1" in message
    assert "diagnostics=" in message


@pytest.mark.parametrize(
    "k,d,seed",
    [(3, 2, 201), (3, 3, 202), (4, 2, 203), (4, 3, 204)],
)
def test_gradient_matches_normalisation_preserving_central_difference(k, d, seed):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-2.0, 2.0, size=(k, d))
    covariances = np.stack([random_covariance(rng, dim=d) for _ in range(k)])
    matrices = np.linalg.inv(covariances)
    center_direction = rng.standard_normal(centers.shape)
    matrix_direction = rng.standard_normal(matrices.shape)
    matrix_direction = 0.5 * (matrix_direction + np.swapaxes(matrix_direction, -1, -2))

    def packed_at(step):
        shifted_centers = centers + step * center_direction
        shifted_matrices = matrices + step * matrix_direction
        return coef_from_cov(shifted_centers, np.linalg.inv(shifted_matrices))

    solver_kwargs = {
        "method": "fw+brentq+newton",
        "tol": 1e-12,
        "newton_tol": 1e-14,
    }
    coefs = packed_at(0.0)
    gradient = ellphi.tangency_simplex_grad(coefs, **solver_kwargs)
    h = 1e-6
    plus = packed_at(h)
    minus = packed_at(-h)
    finite_difference = (
        ellphi.tangency_simplex(plus, **solver_kwargs).t
        - ellphi.tangency_simplex(minus, **solver_kwargs).t
    ) / (2.0 * h)
    coefficient_direction = (plus - minus) / (2.0 * h)
    analytical = float(np.sum(gradient.dt_dcoef * coefficient_direction))

    error = _relative_error(analytical, finite_difference)
    assert error < 1e-6, f"k={k}, d={d}, relative error={error:.3e}"


def test_pairwise_gradient_identity(solver_backend, rng):
    p, q = random_coef_pair(rng, dim=3)
    gradient = ellphi.tangency_simplex_grad(np.stack((p, q)))
    pairwise = ellphi.tangency_grad(p, q, backend=solver_backend)

    error_p = _relative_error(gradient.dt_dcoef[0], pairwise.dt_dp)
    error_q = _relative_error(gradient.dt_dcoef[1], pairwise.dt_dq)
    assert error_p < 1e-6, f"dt_dp relative error={error_p:.3e}"
    assert error_q < 1e-6, f"dt_dq relative error={error_q:.3e}"


def test_zero_time_gradient_raises():
    coefs = coef_from_cov(np.array([[1.0, -2.0]]), np.eye(2)[None])
    with pytest.raises(ZeroDivisionError, match="t == 0"):
        ellphi.tangency_simplex_grad(coefs)


@pytest.mark.parametrize("api", [ellphi.tangency_simplex, ellphi.tangency_simplex_grad])
def test_unnormalized_constant_is_rejected(api):
    coefs = coef_from_cov(
        np.array([[0.0, 0.0], [2.0, 0.0]]), np.repeat(np.eye(2)[None], 2, axis=0)
    )
    coefs[0, -1] += 1.0

    with pytest.raises(ValueError, match=r"row 0.*coef_from_cov"):
        api(coefs)


@pytest.mark.parametrize("api", [ellphi.tangency_simplex, ellphi.tangency_simplex_grad])
@pytest.mark.parametrize("name", ["active_tol", "weight_tol", "norm_tol"])
@pytest.mark.parametrize("value", [np.nan, -1.0])
def test_simplex_tolerances_must_be_finite_and_nonnegative(api, name, value):
    coefs = coef_from_cov(
        np.array([[0.0, 0.0], [2.0, 0.0]]), np.repeat(np.eye(2)[None], 2, axis=0)
    )

    with pytest.raises(ValueError, match=f"{name} must be finite and non-negative"):
        api(coefs, **{name: value})


def test_public_namedtuple_field_order():
    assert ellphi.SimplexTangencyResult._fields == (
        "t",
        "point",
        "mu",
        "support",
        "active_set",
    )
    assert ellphi.SimplexTangencyGrad._fields == (
        "t",
        "point",
        "mu",
        "dt_dcoef",
        "support",
        "active_set",
    )


def test_public_exports_hide_internal_engine():
    expected = {
        "SimplexTangencyResult",
        "SimplexTangencyGrad",
        "tangency_simplex",
        "tangency_simplex_grad",
    }
    assert expected <= set(ellphi.__all__)
    assert not {
        "MinimaxResult",
        "GradientResult",
        "solve_minimax",
        "solve_minimax_from_coefs",
        "compute_gradient",
    } & set(ellphi.__all__)
    for name in (
        "MinimaxResult",
        "GradientResult",
        "solve_minimax",
        "solve_minimax_from_coefs",
        "compute_gradient",
    ):
        assert not hasattr(ellphi, name)
