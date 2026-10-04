"""Tests for the many-body minimax solver.

Adapted from uda-lab/ellcech ``tests/test_minimax.py`` at commit
82d13e3e174f4903cdcd6adcee1f469348361c91.

Key validation:
  For |σ| = 2, alpha(σ) must equal ellphi.tangency(p, q).t ** 2.
  This cross-validates the Frank-Wolfe solver against ellphi's 1D root-finding,
  which is the established reference implementation for the pairwise case.
"""

from typing import get_args

import numpy as np
import pytest

import ellphi
import ellphi._minimax_python as minimax_mod
from ellphi.geometry import unpack_conic
from ellphi._minimax_python import solve_minimax, solve_minimax_from_coefs

from .factories import random_coef_pair


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spd(rng, d):
    """Return a random d×d SPD matrix."""
    L = rng.standard_normal((d, d))
    return L @ L.T + np.eye(d) * 0.5


def _pairwise_alpha_ellphi(pcoef, qcoef):
    """Ground truth: α({i,j}) = t² from ellphi."""
    return ellphi.tangency(pcoef, qcoef).t ** 2


# ---------------------------------------------------------------------------
# Trivial cases
# ---------------------------------------------------------------------------


class TestTrivialCases:
    def test_empty_simplex_raises(self):
        with pytest.raises(ValueError, match="k=0"):
            solve_minimax(
                np.zeros((0, 2, 2)),
                np.zeros((0, 2)),
            )

    def test_single_point_alpha_zero(self):
        A = np.eye(2)
        x = np.array([1.0, 2.0])
        res = solve_minimax(A[np.newaxis], x[np.newaxis])
        assert res.alpha == pytest.approx(0.0, abs=1e-12)
        np.testing.assert_allclose(res.circumcenter, x)
        assert res.converged
        assert res.n_iter == 0

    def test_single_point_3d(self):
        rng = np.random.default_rng(0)
        A = _make_spd(rng, 3)
        x = rng.standard_normal(3)
        res = solve_minimax(A[np.newaxis], x[np.newaxis])
        assert res.alpha == pytest.approx(0.0, abs=1e-12)

    def test_single_point_offset(self):
        res = solve_minimax(
            np.eye(2)[np.newaxis],
            np.array([[1.0, 2.0]]),
            offsets=np.array([0.75]),
        )
        assert res.alpha == 0.75

    def test_single_point_nan_offset_raises(self):
        with pytest.raises(ValueError, match="offsets must be finite"):
            solve_minimax(
                np.eye(2)[np.newaxis],
                np.zeros((1, 2)),
                offsets=np.array([np.nan]),
            )


# ---------------------------------------------------------------------------
# Isotropic (A_i = I) special cases
# ---------------------------------------------------------------------------


class TestIsotropic:
    def test_two_points_midpoint(self):
        """With A_i = I, x* is the midpoint of two equidistant points."""
        x0 = np.array([-1.0, 0.0])
        x1 = np.array([+1.0, 0.0])
        A = np.eye(2)[np.newaxis].repeat(2, axis=0)
        centers = np.stack([x0, x1])
        res = solve_minimax(A, centers)
        assert res.alpha == pytest.approx(1.0, rel=1e-6)
        np.testing.assert_allclose(res.circumcenter, [0.0, 0.0], atol=1e-6)
        assert res.converged

    def test_two_points_alpha_is_half_sq_distance(self):
        """For A=I, α({i,j}) = ||x̄_i - x̄_j||² / 4 (midpoint equidistant)."""
        rng = np.random.default_rng(42)
        x0, x1 = rng.standard_normal((2, 3))
        A = np.eye(3)[np.newaxis].repeat(2, axis=0)
        centers = np.stack([x0, x1])
        res = solve_minimax(A, centers)
        expected = np.sum((x0 - x1) ** 2) / 4.0
        assert res.alpha == pytest.approx(expected, rel=1e-6)

    def test_three_points_circumcenter_equidistant(self):
        """With A_i = I, x* is the circumcenter (equidistant from all points)."""
        # Equilateral triangle
        c = 2.0 * np.pi / 3.0
        x0 = np.array([1.0, 0.0])
        x1 = np.array([np.cos(c), np.sin(c)])
        x2 = np.array([np.cos(2 * c), np.sin(2 * c)])
        A = np.eye(2)[np.newaxis].repeat(3, axis=0)
        centers = np.stack([x0, x1, x2])
        res = solve_minimax(A, centers, tol=1e-10)
        # All f_i should be equal
        diff = res.circumcenter[np.newaxis] - centers
        f = np.einsum("ki,kij,kj->k", diff, A, diff)
        assert np.max(f) - np.min(f) == pytest.approx(0.0, abs=1e-6)
        assert res.converged

    def test_two_points_with_offset(self):
        matrices = np.repeat(np.eye(2)[np.newaxis], 2, axis=0)
        centers = np.array([[0.0, 0.0], [2.0, 0.0]])

        res = solve_minimax(matrices, centers, offsets=np.array([1.0, 0.0]))

        assert res.converged
        assert res.alpha == pytest.approx(1.5625, abs=1e-12)
        np.testing.assert_allclose(res.circumcenter, [0.75, 0.0], atol=1e-12)


# ---------------------------------------------------------------------------
# Cross-validation with ellphi (pairwise case |σ| = 2)
# ---------------------------------------------------------------------------


class TestCrossValidationEllphi:
    """For |σ| = 2, alpha must match ellphi.tangency(p, q).t**2."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 7, 42])
    def test_pairwise_random_2d(self, seed):
        rng = np.random.default_rng(seed)
        x0, x1 = rng.standard_normal((2, 2))
        cov0 = _make_spd(rng, 2)
        cov1 = _make_spd(rng, 2)
        pcoef = ellphi.coef_from_cov(x0, cov0)[0]
        qcoef = ellphi.coef_from_cov(x1, cov1)[0]
        coefs = np.stack([pcoef, qcoef])

        res = solve_minimax_from_coefs(coefs, tol=1e-10)
        expected = _pairwise_alpha_ellphi(pcoef, qcoef)

        assert res.alpha == pytest.approx(
            expected, rel=1e-5
        ), f"seed={seed}: alpha={res.alpha:.8f}, expected={expected:.8f}"
        assert res.converged, f"seed={seed}: did not converge in {res.n_iter} iters"

    @pytest.mark.parametrize("seed", [0, 1, 5])
    def test_pairwise_random_3d(self, seed):
        rng = np.random.default_rng(seed + 100)
        x0, x1 = rng.standard_normal((2, 3))
        cov0 = _make_spd(rng, 3)
        cov1 = _make_spd(rng, 3)
        pcoef = ellphi.coef_from_cov(x0, cov0)[0]
        qcoef = ellphi.coef_from_cov(x1, cov1)[0]
        coefs = np.stack([pcoef, qcoef])

        res = solve_minimax_from_coefs(coefs, tol=1e-10)
        expected = _pairwise_alpha_ellphi(pcoef, qcoef)

        assert res.alpha == pytest.approx(
            expected, rel=1e-5
        ), f"seed={seed}: alpha={res.alpha:.8f}, expected={expected:.8f}"

    def test_pairwise_identical_ellipses(self):
        """Two identical ellipses: α = 0, x* = center."""
        x = np.array([1.0, 2.0])
        cov = np.diag([0.5, 2.0])
        coef = ellphi.coef_from_cov(x, cov)[0]
        coefs = np.stack([coef, coef])
        res = solve_minimax_from_coefs(coefs, tol=1e-10)
        assert res.alpha == pytest.approx(0.0, abs=1e-7)
        np.testing.assert_allclose(res.circumcenter, x, atol=1e-6)

    def test_pairwise_tangent_point_location(self):
        """The circumcenter x* should satisfy f_i(x*) = f_j(x*) = alpha."""
        rng = np.random.default_rng(99)
        x0, x1 = rng.standard_normal((2, 2))
        cov0 = _make_spd(rng, 2)
        cov1 = _make_spd(rng, 2)
        pcoef = ellphi.coef_from_cov(x0, cov0)[0]
        qcoef = ellphi.coef_from_cov(x1, cov1)[0]
        coefs = np.stack([pcoef, qcoef])

        A_arr, b_arr, _ = unpack_conic(coefs)
        centers = np.stack([-np.linalg.solve(A_arr[i], b_arr[i]) for i in range(2)])

        res = solve_minimax(A_arr, centers, tol=1e-12)
        xstar = res.circumcenter
        diff = xstar[np.newaxis] - centers
        f = np.einsum("ki,kij,kj->k", diff, A_arr, diff)

        assert f[0] == pytest.approx(f[1], rel=1e-5)
        assert f[0] == pytest.approx(res.alpha, rel=1e-5)


# ---------------------------------------------------------------------------
# Higher-dimensional simplex (|σ| ≥ 3)
# ---------------------------------------------------------------------------


class TestHigherOrder:
    def test_triangle_active_set(self):
        """For a centered equilateral triangle, all 3 weights should be active."""
        c = 2.0 * np.pi / 3.0
        pts = np.array(
            [
                [1.0, 0.0],
                [np.cos(c), np.sin(c)],
                [np.cos(2 * c), np.sin(2 * c)],
            ]
        )
        A = np.eye(2)[np.newaxis].repeat(3, axis=0)
        res = solve_minimax(A, pts, tol=1e-10)
        assert len(res.active_set) == 3
        assert res.converged

    def test_degenerate_triangle_reduces_to_edge(self):
        """If one point is much closer, its weight should be near zero (inactive)."""
        # x2 is very close to x* of {x0, x1}, so active set should be {0, 1}
        pts = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1e-6]])
        A = np.eye(2)[np.newaxis].repeat(3, axis=0)
        res = solve_minimax(A, pts, tol=1e-8)
        # alpha should be close to alpha({0, 1}) = 1.0
        assert res.alpha == pytest.approx(1.0, rel=1e-3)

    @pytest.mark.parametrize("k,d", [(3, 2), (4, 3), (5, 4)])
    def test_random_simplex_convergence(self, k, d):
        """Frank-Wolfe should converge for random k-simplex in R^d."""
        rng = np.random.default_rng(k * 100 + d)
        centers = rng.standard_normal((k, d))
        matrices = np.stack([_make_spd(rng, d) for _ in range(k)])
        res = solve_minimax(matrices, centers, tol=1e-8, max_iter=5000)
        assert res.converged, f"k={k}, d={d}: did not converge in {res.n_iter} iters"
        assert res.alpha >= 0.0
        assert len(res.active_set) >= 1


# ---------------------------------------------------------------------------
# Numerical stability controls (Gap F)
# ---------------------------------------------------------------------------


class TestNumericalStabilityControls:
    @pytest.mark.parametrize("method", list(get_args(minimax_mod.MethodName)))
    def test_zero_offsets_are_bitwise_identical(self, method):
        matrices = np.repeat(np.eye(2)[np.newaxis], 3, axis=0)
        centers = np.array([[0.0, 0.0], [2.0, 0.0], [0.5, 1.0]])

        baseline = solve_minimax(matrices, centers, method=method)
        explicit_zeros = solve_minimax(
            matrices, centers, offsets=np.zeros(3), method=method
        )

        assert explicit_zeros.alpha == baseline.alpha
        np.testing.assert_array_equal(
            explicit_zeros.circumcenter, baseline.circumcenter
        )
        np.testing.assert_array_equal(explicit_zeros.weights, baseline.weights)
        assert explicit_zeros.converged == baseline.converged

        explicit_stability_zeros = solve_minimax(
            matrices,
            centers,
            method=method,
            regularization=0.0,
            condition_number_limit=None,
            max_conditioning_steps=8,
        )
        assert explicit_stability_zeros.alpha == baseline.alpha
        np.testing.assert_array_equal(
            explicit_stability_zeros.circumcenter, baseline.circumcenter
        )
        np.testing.assert_array_equal(
            explicit_stability_zeros.weights, baseline.weights
        )
        assert explicit_stability_zeros.converged == baseline.converged

    def test_explicit_default_controls_match_previous_behaviour(self):
        rng = np.random.default_rng(321)
        centers = rng.standard_normal((4, 3))
        matrices = np.stack([_make_spd(rng, 3) for _ in range(4)])

        baseline = solve_minimax(matrices, centers, tol=1e-10)
        with_explicit_defaults = solve_minimax(
            matrices,
            centers,
            tol=1e-10,
            regularization=0.0,
            condition_number_limit=None,
            max_conditioning_steps=8,
        )

        assert with_explicit_defaults.alpha == pytest.approx(
            baseline.alpha, rel=1e-12, abs=1e-12
        )
        np.testing.assert_allclose(
            with_explicit_defaults.circumcenter,
            baseline.circumcenter,
            atol=1e-12,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            with_explicit_defaults.weights,
            baseline.weights,
            atol=1e-12,
            rtol=1e-12,
        )

    def test_regularization_and_conditioning_produce_finite_solution(self):
        matrices = np.array(
            [
                [[1e-12, 0.0], [0.0, 1.0]],
                [[1e12, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [0.0, 1e-12]],
            ]
        )
        centers = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
            ]
        )

        res = solve_minimax(
            matrices,
            centers,
            regularization=1e-8,
            condition_number_limit=1e8,
            max_conditioning_steps=4,
            tol=1e-9,
            max_iter=4000,
        )
        assert np.isfinite(res.alpha)
        assert np.all(np.isfinite(res.circumcenter))
        assert np.all(np.isfinite(res.weights))
        assert np.sum(res.weights) == pytest.approx(1.0, abs=1e-8)

    def test_stability_control_parameter_validation(self):
        A = np.eye(2)[np.newaxis].repeat(2, axis=0)
        centers = np.array([[0.0, 0.0], [1.0, 0.0]])
        with pytest.raises(ValueError):
            solve_minimax(A, centers, regularization=-1.0)
        with pytest.raises(ValueError):
            solve_minimax(A, centers, condition_number_limit=1.0)
        with pytest.raises(ValueError):
            solve_minimax(A, centers, max_conditioning_steps=-1)
        with pytest.raises(ValueError, match="offsets shape"):
            solve_minimax(A, centers, offsets=np.zeros(3))

    @pytest.mark.parametrize("value", [np.inf, np.nan, 0.0])
    def test_fw_tolerance_must_be_finite_and_positive(self, value):
        A = np.eye(2)[np.newaxis].repeat(2, axis=0)
        centers = np.array([[0.0, 0.0], [1.0, 0.0]])
        with pytest.raises(ValueError, match="tol must be finite and > 0"):
            solve_minimax(A, centers, tol=value)

    @pytest.mark.parametrize("k", [2, 3, 4])
    @pytest.mark.parametrize("weight_tol", [None, 0.5])
    def test_weight_tol_cannot_empty_the_fw_face(self, k, weight_tol):
        A = np.repeat(np.eye(2)[np.newaxis], k, axis=0)
        centers = np.arange(2 * k, dtype=float).reshape(k, 2)
        threshold = 1.0 / k if weight_tol is None else weight_tol
        with pytest.raises(ValueError, match="weight_tol must satisfy"):
            solve_minimax(A, centers, weight_tol=threshold)

    def test_fw_empty_face_fallback_does_not_false_converge(self):
        A = np.repeat(np.eye(2)[np.newaxis], 2, axis=0)
        centers = np.array([[0.0, 0.0], [2.0, 0.0]])
        offsets = np.array([1.0, 0.0])
        Ax = np.einsum("kij,kj->ki", A, centers)
        mu, converged, n_iter = minimax_mod._run_fw_bisect(
            A,
            Ax,
            centers,
            offsets,
            np.full(2, 0.5),
            tol=1e-12,
            max_iter=1,
            weight_tol=1.0,
            regularization=0.0,
            condition_number_limit=None,
            max_conditioning_steps=8,
        )

        assert n_iter == 1
        assert not converged
        assert np.all(np.isfinite(mu))

    def test_regularized_slsqp_objective_gradient_matches(self):
        matrices = np.array(
            [
                [[2.0, 0.2], [0.2, 1.0]],
                [[1.0, -0.1], [-0.1, 3.0]],
                [[1.5, 0.3], [0.3, 2.0]],
            ]
        )
        centers = np.array([[0.0, 0.0], [1.0, -0.5], [-0.5, 1.0]])
        Ax = np.einsum("kij,kj->ki", matrices, centers)
        mu = np.array([0.2, 0.3, 0.5])
        objective, gradient = minimax_mod._slsqp_objective_and_gradient(
            mu,
            matrices,
            Ax,
            centers,
            None,
            regularization=0.25,
            condition_number_limit=None,
            max_conditioning_steps=8,
        )
        step = 1e-6
        finite_difference = np.empty_like(mu)
        for i in range(mu.size):
            plus = mu.copy()
            minus = mu.copy()
            plus[i] += step
            minus[i] -= step
            plus_value, _ = minimax_mod._slsqp_objective_and_gradient(
                plus,
                matrices,
                Ax,
                centers,
                None,
                regularization=0.25,
                condition_number_limit=None,
                max_conditioning_steps=8,
            )
            minus_value, _ = minimax_mod._slsqp_objective_and_gradient(
                minus,
                matrices,
                Ax,
                centers,
                None,
                regularization=0.25,
                condition_number_limit=None,
                max_conditioning_steps=8,
            )
            finite_difference[i] = (plus_value - minus_value) / (2.0 * step)

        np.testing.assert_allclose(finite_difference, gradient, rtol=1e-6, atol=1e-8)
        assert np.isfinite(objective)

    def test_conditioning_slsqp_objective_gradient_matches(self):
        matrices = np.array(
            [
                [[1e-8, 0.0], [0.0, 1.0]],
                [[2e-8, 0.0], [0.0, 1.0]],
                [[3e-8, 0.0], [0.0, 1.0]],
            ]
        )
        centers = np.array([[0.0, 0.0], [1.0, -0.5], [-0.5, 1.0]])
        Ax = np.einsum("kij,kj->ki", matrices, centers)
        mu = np.array([0.2, 0.3, 0.5])
        kwargs = {
            "regularization": 0.0,
            "condition_number_limit": 1e4,
            "max_conditioning_steps": 12,
        }
        objective, gradient = minimax_mod._slsqp_objective_and_gradient(
            mu, matrices, Ax, centers, None, **kwargs
        )
        step = 1e-6
        finite_difference = np.empty_like(mu)
        for i in range(mu.size):
            plus = mu.copy()
            minus = mu.copy()
            plus[i] += step
            minus[i] -= step
            plus_value, _ = minimax_mod._slsqp_objective_and_gradient(
                plus, matrices, Ax, centers, None, **kwargs
            )
            minus_value, _ = minimax_mod._slsqp_objective_and_gradient(
                minus, matrices, Ax, centers, None, **kwargs
            )
            finite_difference[i] = (plus_value - minus_value) / (2.0 * step)

        np.testing.assert_allclose(finite_difference, gradient, rtol=1e-6, atol=1e-8)
        assert np.isfinite(objective)

    def test_newton_polishing_uses_stabilized_stationarity(self):
        matrices = np.array([np.eye(2), 4.0 * np.eye(2)])
        centers = np.array([[0.0, 0.0], [1.0, 0.0]])
        stabilized = solve_minimax(
            matrices,
            centers,
            method="fw+bisect+newton",
            regularization=1.0,
            tol=1e-12,
        )
        unadjusted = solve_minimax(
            matrices, centers, method="fw+bisect+newton", tol=1e-12
        )
        Ax = np.einsum("kij,kj->ki", matrices, centers)
        _, values = minimax_mod._eval_f(
            stabilized.weights,
            matrices,
            Ax,
            centers,
            None,
            regularization=1.0,
            condition_number_limit=None,
            max_conditioning_steps=8,
        )

        assert stabilized.converged
        assert np.max(values) - stabilized.weights @ values < 1e-12
        assert stabilized.alpha == pytest.approx(4.0 / 9.0, abs=1e-12)
        assert unadjusted.alpha == pytest.approx(stabilized.alpha, abs=1e-12)
        assert not np.array_equal(stabilized.weights, unadjusted.weights)

    def test_fw_gap_ignores_subthreshold_positive_weights(self):
        matrices = np.repeat(np.eye(2)[np.newaxis], 3, axis=0)
        centers = np.array([[2.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
        Ax = np.einsum("kij,kj->ki", matrices, centers)
        mu = np.array([0.405, 0.405, 0.19])

        _, converged, n_iter = minimax_mod._run_fw_bisect(
            matrices,
            Ax,
            centers,
            None,
            mu,
            tol=1e-12,
            max_iter=1,
            weight_tol=0.2,
            regularization=0.0,
            condition_number_limit=None,
            max_conditioning_steps=8,
        )

        assert n_iter == 1
        assert not converged

    def test_slsqp_converges_with_regularization(self):
        matrices = np.array(
            [
                [[2.0, 0.2], [0.2, 1.0]],
                [[1.0, -0.1], [-0.1, 3.0]],
                [[1.5, 0.3], [0.3, 2.0]],
            ]
        )
        centers = np.array([[0.0, 0.0], [1.0, -0.5], [-0.5, 1.0]])
        result = solve_minimax(
            matrices,
            centers,
            method="scipy-slsqp",
            regularization=0.25,
        )

        assert result.converged
        assert np.isfinite(result.alpha)
        assert np.all(np.isfinite(result.weights))
        assert result.weights.sum() == pytest.approx(1.0)

    def test_slsqp_zero_regularization_objective_path_is_unchanged(self):
        matrices = np.repeat(np.eye(2)[np.newaxis], 2, axis=0)
        centers = np.array([[0.0, 0.0], [2.0, 0.0]])
        Ax = np.einsum("kij,kj->ki", matrices, centers)
        mu = np.array([0.25, 0.75])
        objective, gradient = minimax_mod._slsqp_objective_and_gradient(
            mu,
            matrices,
            Ax,
            centers,
            None,
            regularization=0.0,
            condition_number_limit=None,
            max_conditioning_steps=8,
        )
        _, f = minimax_mod._eval_f(
            mu,
            matrices,
            Ax,
            centers,
            None,
            regularization=0.0,
            condition_number_limit=None,
            max_conditioning_steps=8,
        )

        assert objective == -float(np.dot(mu, f))
        np.testing.assert_array_equal(gradient, -f)

    def test_conditioning_steps_cap_is_strict(self):
        A = np.diag([1e-12, 1.0])
        eye = np.eye(2)
        limited0 = minimax_mod._condition_matrix(
            A,
            regularization=1e-6,
            condition_number_limit=2.0,
            max_conditioning_steps=0,
        )
        limited1 = minimax_mod._condition_matrix(
            A,
            regularization=1e-6,
            condition_number_limit=2.0,
            max_conditioning_steps=1,
        )

        np.testing.assert_allclose(limited0, A + 1e-6 * eye)
        np.testing.assert_allclose(limited1, A + 1e-5 * eye)

    def test_from_coefs_matches_direct_minimax_with_non_default_stability(self):
        rng = np.random.default_rng(77)
        x0, x1 = rng.standard_normal((2, 2))
        cov0 = _make_spd(rng, 2)
        cov1 = _make_spd(rng, 2)
        pcoef = ellphi.coef_from_cov(x0, cov0)[0]
        qcoef = ellphi.coef_from_cov(x1, cov1)[0]
        coefs = np.stack([pcoef, qcoef])

        A_arr, b_arr, _ = unpack_conic(coefs)
        centers = np.stack([-np.linalg.solve(A_arr[i], b_arr[i]) for i in range(2)])

        kwargs = {
            "regularization": 1e-6,
            "condition_number_limit": 1e6,
            "max_conditioning_steps": 2,
            "tol": 1e-10,
        }
        direct = solve_minimax(A_arr, centers, **kwargs)
        from_coefs = solve_minimax_from_coefs(coefs, **kwargs)

        assert from_coefs.alpha == pytest.approx(direct.alpha, rel=1e-10, abs=1e-12)
        np.testing.assert_allclose(
            from_coefs.circumcenter,
            direct.circumcenter,
            rtol=1e-10,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            from_coefs.weights, direct.weights, rtol=1e-10, atol=1e-12
        )


# ---------------------------------------------------------------------------
# Pairwise compatibility with ellphi.tangency (both backends)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim", [2, 3])
def test_pairwise_alpha_is_squared_tangency_time(solver_backend, rng, dim):
    """For k = 2, ``solve_minimax(...).alpha == tangency(p, q).t ** 2``."""
    for _ in range(5):
        pcoef, qcoef = random_coef_pair(rng, dim=dim)
        t = ellphi.tangency(pcoef, qcoef, backend=solver_backend).t

        A_arr, b_arr, _ = unpack_conic(np.stack([pcoef, qcoef]))
        centers = np.stack([-np.linalg.solve(A_arr[i], b_arr[i]) for i in range(2)])
        res = solve_minimax(A_arr, centers)

        assert res.converged
        assert res.alpha == pytest.approx(t**2, rel=1e-9)
        assert res.weights.sum() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("dim", [2, 3])
def test_unnormalized_pairwise_coefs_match_tangency(solver_backend, rng, dim):
    """General packed constants agree with pairwise tangency."""
    for _ in range(5):
        pcoef, qcoef = random_coef_pair(rng, dim=dim)
        coefs = np.stack([pcoef, qcoef])
        coefs[:, -1] += rng.uniform(0.1, 1.0, size=2)

        pairwise = ellphi.tangency(coefs[0], coefs[1], backend=solver_backend)
        res = solve_minimax_from_coefs(coefs, tol=1e-11)

        assert res.converged
        assert res.alpha == pytest.approx(pairwise.t**2, rel=1e-9)
        np.testing.assert_allclose(
            res.circumcenter, pairwise.point, rtol=1e-9, atol=1e-10
        )


# ---------------------------------------------------------------------------
# Regression fixture: weight support versus tight constraint set
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", list(get_args(minimax_mod.MethodName)))
def test_right_triangle_support_differs_from_tight_constraint_set(method):
    """Support-vs-tight-set fixture: three unit balls on a right triangle.

    The centers ``(2, 0)``, ``(0, 2)`` and ``(0, 0)`` give ``alpha = 2`` at the
    circumcenter ``(1, 1)``. All three constraints are tight there, but the
    unique optimal weights are ``(1/2, 1/2, 0)``, so ``active_set`` (the
    weight support) has two elements while the tight set has three.
    """
    centers = np.array([[2.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
    matrices = np.repeat(np.eye(2)[np.newaxis], 3, axis=0)

    res = solve_minimax(matrices, centers, method=method)

    assert res.converged
    assert res.alpha == pytest.approx(2.0, rel=1e-12)
    np.testing.assert_allclose(res.circumcenter, [1.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(res.weights, [0.5, 0.5, 0.0], atol=1e-10)
    assert res.active_set == [0, 1]

    diff = res.circumcenter[np.newaxis] - centers
    f = np.einsum("ki,kij,kj->k", diff, matrices, diff)
    tight = [i for i in range(3) if abs(f[i] - res.alpha) <= 1e-12 * res.alpha]
    assert tight == [0, 1, 2]
