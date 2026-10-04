"""Tests for the anisotropic Čech filtration API."""

from __future__ import annotations

import importlib
from typing import get_args

import numpy as np
import pytest

import ellphi
from ellphi._minimax_python import MethodName, solve_minimax
from ellphi.geometry import coef_from_cov, pack_conic, unpack_conic

from .factories import minimax_public_surrogate, random_coef_pair, random_covariance


cech_module = importlib.import_module("ellphi.cech")


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


def _value_roundoff_bound(coefs: np.ndarray) -> float:
    matrices, linear, constants = unpack_conic(coefs)
    return sum(
        64.0
        * np.finfo(float).eps
        * np.linalg.cond(matrix, 2)
        * max(
            abs(float(constant)),
            abs(float(vector @ np.linalg.solve(matrix, vector))),
        )
        for matrix, vector, constant in zip(matrices, linear, constants)
    )


def test_pairwise_cech_agrees_with_tangency(solver_backend, rng):
    p, q = random_coef_pair(rng, dim=3)
    result = ellphi.cech(np.stack((p, q)))
    pairwise = ellphi.tangency(p, q, backend=solver_backend)

    assert result.t == pytest.approx(pairwise.t, rel=1e-9)
    np.testing.assert_allclose(result.point, pairwise.point, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(
        result.mu, (1.0 - pairwise.mu, pairwise.mu), rtol=1e-8, atol=1e-10
    )


@pytest.mark.parametrize("dimension", [2, 3])
def test_unnormalized_pairwise_cech_agrees_with_tangency(
    solver_backend, rng, dimension
):
    for _ in range(5):
        centers = rng.uniform(-2.0, 2.0, size=(2, dimension))
        while np.linalg.norm(centers[0] - centers[1]) < 1.5:
            centers = rng.uniform(-2.0, 2.0, size=(2, dimension))
        covariances = np.stack(
            [random_covariance(rng, dim=dimension) for _ in range(2)]
        )
        coefs = coef_from_cov(centers, covariances)
        coefs[:, -1] += rng.uniform(0.1, 1.0) + rng.uniform(-1e-4, 1e-4, size=2)

        result = ellphi.cech(coefs)
        pairwise = ellphi.tangency(coefs[0], coefs[1], backend=solver_backend)

        assert result.t == pytest.approx(pairwise.t, rel=1e-9)
        np.testing.assert_allclose(result.point, pairwise.point, rtol=1e-9, atol=1e-10)


def test_translated_pair_uses_engine_value_and_centred_constraints(solver_backend):
    centers = np.array([[1e6, 0.0], [1e6 + 2.0, 0.0]])
    coefs = coef_from_cov(centers, np.repeat(np.eye(2)[None], 2, axis=0))
    pairwise_coefs = coef_from_cov(
        np.array([[0.0, 0.0], [2.0, 0.0]]), np.repeat(np.eye(2)[None], 2, axis=0)
    )

    # Packed inputs carry roundoff of order eps*|xbar|**2; this is a
    # representation limit, not a solver property.
    matrices, _, constants = unpack_conic(coefs)
    translation_bound = max(
        64.0
        * np.finfo(float).eps
        * np.linalg.cond(matrix, 2)
        * max(1.0, abs(float(constant)))
        for matrix, constant in zip(matrices, constants)
    )
    result = ellphi.cech(coefs)
    pairwise = ellphi.tangency(
        pairwise_coefs[0], pairwise_coefs[1], backend=solver_backend
    )
    gradient = ellphi.cech_grad(coefs)
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

    assert abs(result.t**2 - pairwise.t**2) <= translation_bound
    assert result.active_set == (0, 1)
    assert np.all(np.isfinite(gradient.dt_dcoef))
    assert _relative_error(gradient.dt_dcoef[0], pairwise_identity) < 1e-6
    assert _relative_error(gradient.dt_dcoef[1], pairwise_identity) < 1e-6


def test_translated_singleton_roundoff_negative_scale_is_clipped():
    center = np.array([[1e6, -1e6]])
    covariance = np.array([[1.0, 0.2], [0.2, 1.7]])
    coefs = coef_from_cov(center, covariance[None])
    value_roundoff = _value_roundoff_bound(coefs)

    result = ellphi.cech(coefs)

    assert abs(result.t) <= np.sqrt(value_roundoff)


def test_translated_coincident_roundoff_negative_scale_is_clipped():
    center = np.array([[1e6, -1e6]])
    covariance = np.array([[1.0, 0.2], [0.2, 1.7]])
    coefs = np.repeat(coef_from_cov(center, covariance[None]), 2, axis=0)
    value_roundoff = _value_roundoff_bound(coefs)

    result = ellphi.cech(coefs)

    assert abs(result.t) <= np.sqrt(value_roundoff)


@pytest.mark.parametrize("dimension", [2, 3, 4])
@pytest.mark.parametrize("condition_number", [1e2, 1e6, 1e10, 1e12])
def test_coef_from_cov_accepts_rotated_spd_conditioning(dimension, condition_number):
    rng = np.random.default_rng(2700 + dimension)
    rotation, _ = np.linalg.qr(rng.standard_normal((dimension, dimension)))
    eigenvalues = np.geomspace(1.0, condition_number, dimension)
    covariance = rotation @ np.diag(eigenvalues) @ rotation.T
    center = rng.uniform(-2.0, 2.0, size=(1, dimension))
    coefs = coef_from_cov(center, covariance[None])

    matrices, linear, constants = unpack_conic(coefs)
    precision_bound = (
        64.0
        * np.finfo(float).eps
        * np.linalg.cond(matrices[0], 2)
        * max(1.0, abs(float(constants[0])))
    )

    if condition_number <= 1e6:
        result = ellphi.cech(coefs)
        assert result.t**2 <= precision_bound
        point = result.point
    else:
        inverse_times_linear = np.linalg.solve(matrices[0], linear[0])
        centers = np.stack([-inverse_times_linear])
        centered_constant = float(linear[0] @ inverse_times_linear)
        result = solve_minimax(
            matrices, centers, offsets=np.array([constants[0] - centered_constant])
        )
        assert abs(result.alpha) <= precision_bound
        point = result.circumcenter
    assert np.all(np.isfinite(point))


def test_scaled_centres_from_cov_are_accepted():
    centers = np.array([[1e6, -1e6], [1e6 + 2.0, -1e6 + 1.0]])
    covariances = np.array(
        [
            [[2.0, 0.3], [0.3, 1.5]],
            [[1.2, -0.2], [-0.2, 0.8]],
        ]
    )

    result = ellphi.cech(coef_from_cov(centers, covariances))

    assert np.isfinite(result.t)
    assert result.active_set == (0, 1)


def test_subthreshold_packed_constant_is_honoured_by_gradient():
    matrices = np.repeat(np.eye(2)[np.newaxis], 2, axis=0)
    centers = np.array([[0.0, 0.0], [2.0, 0.0]])
    linear = -np.einsum("kij,kj->ki", matrices, centers)
    baseline_coefs = pack_conic(matrices, linear, np.array([0.0, 4.0]))
    perturbed_coefs = baseline_coefs.copy()
    perturbed_coefs[0, -1] += 5e-10
    solver_kwargs = {"method": "fw+brentq+newton", "tol": 1e-12}

    baseline = ellphi.cech(baseline_coefs, **solver_kwargs)
    perturbed = ellphi.cech(perturbed_coefs, **solver_kwargs)
    gradient = ellphi.cech_grad(perturbed_coefs, **solver_kwargs)

    expected_change = baseline.mu[0] * 5e-10
    assert perturbed.t**2 - baseline.t**2 == pytest.approx(
        expected_change, rel=1e-6, abs=1e-15
    )

    h = 1e-7
    plus = perturbed_coefs.copy()
    minus = perturbed_coefs.copy()
    plus[0, -1] += h
    minus[0, -1] -= h
    finite_difference = (
        ellphi.cech(plus, **solver_kwargs).t - ellphi.cech(minus, **solver_kwargs).t
    ) / (2.0 * h)
    assert gradient.dt_dcoef[0, -1] == pytest.approx(
        finite_difference, rel=1e-6, abs=1e-12
    )


def test_negative_packed_scale_raises():
    coefs = coef_from_cov(
        np.array([[0.0, 0.0], [2.0, 0.0]]), np.repeat(np.eye(2)[None], 2, axis=0)
    )
    coefs[:, -1] -= 2.0

    with pytest.raises(
        ValueError, match="packed quadrics have no common non-negative filtration scale"
    ):
        ellphi.cech(coefs)


@pytest.mark.parametrize("api", [ellphi.cech, ellphi.cech_grad])
@pytest.mark.parametrize(
    "matrix",
    [
        np.diag([1.0, -1.0]),
        -np.eye(2),
    ],
)
def test_public_api_rejects_non_spd_quadratic_blocks(api, matrix):
    coefs = pack_conic(matrix[None], np.zeros((1, 2)), np.zeros(1))

    with pytest.raises(
        ValueError,
        match=r"coefficient row 0 has a non-positive definite quadratic matrix",
    ):
        api(coefs)


@pytest.mark.parametrize("api", [ellphi.cech, ellphi.cech_grad])
def test_public_api_rejects_non_symmetric_quadratic_blocks(api, monkeypatch):
    matrices = np.array([[[2.0, 0.25], [0.0, 2.0]]])
    linear = np.zeros((1, 2))
    constants = np.zeros(1)
    monkeypatch.setattr(
        cech_module,
        "unpack_conic",
        lambda _: (matrices, linear, constants),
    )

    with pytest.raises(
        ValueError,
        match=r"coefficient row 0 has a non-symmetric quadratic matrix",
    ):
        api(np.zeros((1, 6)))


@pytest.mark.parametrize("api", [ellphi.cech, ellphi.cech_grad])
def test_public_api_accepts_ill_conditioned_spd_quadratic_block(api):
    matrix = np.diag([1.0, 1e12])
    matrices = np.repeat(matrix[None], 2, axis=0)
    centers = np.array([[0.0, 0.0], [2.0, 0.0]])
    linear = -np.einsum("kij,kj->ki", matrices, centers)
    constants = np.einsum("ki,kij,kj->k", centers, matrices, centers)
    coefs = pack_conic(matrices, linear, constants)

    result = api(coefs)
    assert result.t == pytest.approx(1.0)


def test_packed_one_dimensional_input_is_rejected_like_pairwise():
    coefs = np.array([[1.0, -2.0, 1.0], [1.0, -4.0, 4.0]])

    with pytest.raises(
        ValueError, match="packed input requires d >= 2, as for ellphi.tangency"
    ):
        ellphi.cech(coefs)


@pytest.mark.parametrize("method", list(get_args(MethodName)))
def test_right_triangle_distinguishes_support_and_active_set(method):
    centers = np.array([[2.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
    coefs = coef_from_cov(centers, np.repeat(np.eye(2)[None], 3, axis=0))

    result = ellphi.cech(coefs, method=method, active_tol=1e-12)

    assert result.t == pytest.approx(np.sqrt(2.0), rel=1e-12)
    np.testing.assert_allclose(result.point, [1.0, 1.0], atol=1e-12)
    assert len(result.support) == 2
    assert len(result.active_set) == 3
    assert result.support == (0, 1)
    assert set(result.support) <= set(result.active_set)
    assert result.active_set == (0, 1, 2)


def test_singleton_and_empty_cech_input():
    coefs = coef_from_cov(np.array([[1.0, -2.0]]), np.eye(2)[None])
    result = ellphi.cech(coefs)

    assert result.t == 0.0
    np.testing.assert_allclose(result.point, [1.0, -2.0])
    np.testing.assert_array_equal(result.mu, [1.0])
    assert result.support == (0,)
    assert result.active_set == (0,)

    with pytest.raises(ValueError, match="at least one vertex"):
        ellphi.cech(np.empty((0, coefs.shape[1])))


def test_non_convergence_raises_with_diagnostics():
    coefs = _random_coefs(5, 3, seed=101)
    with pytest.raises(RuntimeError) as exc_info:
        ellphi.cech(coefs, method="fw+bisect", max_iter=1, tol=1e-15)

    message = str(exc_info.value)
    assert "method='fw+bisect'" in message
    assert "n_iter=1" in message
    assert "diagnostics=" in message


@pytest.mark.parametrize("api", [ellphi.cech, ellphi.cech_grad])
@pytest.mark.parametrize(
    "name, value",
    [
        ("regularization", 1e-6),
        ("condition_number_limit", 1e6),
        ("max_conditioning_steps", 4),
    ],
)
def test_public_api_rejects_private_stabilization_controls(api, name, value):
    coefs = _random_coefs(2, 2, seed=107)

    with pytest.raises(TypeError):
        api(coefs, **{name: value})


def test_public_newton_cold_returns_a_successful_fw_fallback():
    coefs = _random_coefs(5, 3, seed=0)

    result = ellphi.cech(
        coefs,
        method="newton-cold",
        newton_max_iter=1,
        tol=1e-10,
    )

    assert np.isfinite(result.t)
    assert result.active_set


@pytest.mark.parametrize("k,d,seed", [(3, 2, 201), (4, 2, 203)])
def test_gradient_matches_independent_packed_central_difference(k, d, seed):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-2.0, 2.0, size=(k, d))
    covariances = np.stack([random_covariance(rng, dim=d) for _ in range(k)])
    coefs = coef_from_cov(centers, covariances)
    coefs[:, -1] += rng.uniform(0.1, 1.0, size=k)

    solver_kwargs = {
        "method": "fw+brentq+newton",
        "tol": 1e-12,
        "newton_tol": 1e-14,
    }
    gradient = ellphi.cech_grad(coefs, **solver_kwargs)
    h = 1e-6
    finite_difference = np.empty_like(coefs)
    for index in np.ndindex(coefs.shape):
        plus = coefs.copy()
        minus = coefs.copy()
        plus[index] += h
        minus[index] -= h
        finite_difference[index] = (
            ellphi.cech(plus, **solver_kwargs).t - ellphi.cech(minus, **solver_kwargs).t
        ) / (2.0 * h)

    error = _relative_error(gradient.dt_dcoef, finite_difference)
    assert error < 1e-6, f"k={k}, d={d}, relative error={error:.3e}"


def test_pairwise_gradient_identity(solver_backend, rng):
    p, q = random_coef_pair(rng, dim=3)
    gradient = ellphi.cech_grad(np.stack((p, q)))
    pairwise = ellphi.tangency_grad(p, q, backend=solver_backend)

    error_p = _relative_error(gradient.dt_dcoef[0], pairwise.dt_dp)
    error_q = _relative_error(gradient.dt_dcoef[1], pairwise.dt_dq)
    assert error_p < 1e-6, f"dt_dp relative error={error_p:.3e}"
    assert error_q < 1e-6, f"dt_dq relative error={error_q:.3e}"


def test_zero_time_gradient_raises():
    coefs = coef_from_cov(np.array([[1.0, -2.0]]), np.eye(2)[None])
    with pytest.raises(ZeroDivisionError, match="t == 0"):
        ellphi.cech_grad(coefs)


@pytest.mark.parametrize("api", [ellphi.cech, ellphi.cech_grad])
def test_unnormalized_constant_agrees_with_expected(api):
    coefs = coef_from_cov(
        np.array([[0.0, 0.0], [2.0, 0.0]]), np.repeat(np.eye(2)[None], 2, axis=0)
    )
    coefs[0, -1] += 1.0

    result = api(coefs)

    assert result.t == pytest.approx(1.25)
    np.testing.assert_allclose(result.point, [0.75, 0.0], atol=1e-12)
    assert result.active_set == (0, 1)


@pytest.mark.parametrize("method", list(get_args(MethodName)))
def test_public_surrogate_methods(method):
    matrices, centers = minimax_public_surrogate()
    linear = -np.einsum("kij,kj->ki", matrices, centers)
    constants = np.einsum("ki,kij,kj->k", centers, matrices, centers)
    coefs = pack_conic(matrices, linear, constants)

    if method == "scipy-slsqp":
        result = ellphi.cech(coefs, method=method)
        assert result.t == pytest.approx(np.sqrt(0.9024444260258915), abs=1e-8)
    else:
        with pytest.raises(RuntimeError):
            ellphi.cech(coefs, method=method)


@pytest.mark.parametrize("api", [ellphi.cech, ellphi.cech_grad])
@pytest.mark.parametrize("name", ["active_tol", "weight_tol"])
@pytest.mark.parametrize("value", [np.nan, -1.0])
def test_cech_tolerances_must_be_finite_and_nonnegative(api, name, value):
    coefs = coef_from_cov(
        np.array([[0.0, 0.0], [2.0, 0.0]]), np.repeat(np.eye(2)[None], 2, axis=0)
    )

    with pytest.raises(ValueError, match=f"{name} must be finite and non-negative"):
        api(coefs, **{name: value})


@pytest.mark.parametrize("value", [np.inf, np.nan, 0.0])
def test_public_tol_must_be_finite_and_positive(value):
    coefs = coef_from_cov(
        np.array([[0.0, 0.0], [2.0, 0.0]]), np.repeat(np.eye(2)[None], 2, axis=0)
    )

    with pytest.raises(ValueError, match="tol must be finite and > 0"):
        ellphi.cech(coefs, tol=value)


@pytest.mark.parametrize("value", [np.nan, 0.0])
def test_public_newton_tol_must_be_finite_and_positive(value):
    coefs = coef_from_cov(
        np.array([[0.0, 0.0], [2.0, 0.0]]), np.repeat(np.eye(2)[None], 2, axis=0)
    )

    with pytest.raises(ValueError, match="newton_tol must be finite and > 0"):
        ellphi.cech(coefs, newton_tol=value)


def test_public_namedtuple_field_order():
    assert ellphi.CechResult._fields == (
        "t",
        "point",
        "mu",
        "support",
        "active_set",
    )
    assert ellphi.CechGrad._fields == (
        "t",
        "point",
        "mu",
        "dt_dcoef",
        "support",
        "active_set",
    )


def test_public_exports_hide_internal_engine():
    expected = {
        "CechResult",
        "CechGrad",
        "cech",
        "cech_grad",
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
