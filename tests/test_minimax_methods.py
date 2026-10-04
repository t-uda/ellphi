"""Tests for multi-method dispatch in the minimax solver.

Adapted from uda-lab/ellcech ``tests/test_minimax_methods.py`` at commit
82d13e3e174f4903cdcd6adcee1f469348361c91. The benchmark smoke tests are not
ported because the benchmark module is not part of ellphi.

All methods must agree on alpha to tight tolerances.
fw+bisect is the reference (validated against ellphi in test_minimax.py).
"""

from __future__ import annotations

import warnings
from typing import get_args

import numpy as np
import pytest

from ellphi._minimax_python import MethodName, MinimaxResult, solve_minimax

from .factories import minimax_public_surrogate, random_simplex


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_simplex(k: int, d: int, seed: int = 0):
    return random_simplex(k, d, rng=np.random.default_rng(seed))


ALL_METHODS: list[str] = list(get_args(MethodName))

CANONICAL_METHODS = (
    "fw+bisect",
    "fw+brentq",
    "fw+bisect+newton",
    "fw+brentq+newton",
    "fw+bisect+damped-newton",
    "scipy-slsqp",
    "newton-cold",
)

SURROGATE_ALPHA = 0.9024444260258915


def _surrogate_diagnostics(result, matrices, centers):
    diff = result.circumcenter[np.newaxis, :] - centers
    values = np.einsum("ki,kij,kj->k", diff, matrices, diff)
    gap = float(np.max(values) - result.weights @ values)
    stationarity = np.einsum("k,kij,kj->i", result.weights, matrices, diff)
    residual = float(np.linalg.norm(stationarity, ord=np.inf))
    return gap, residual


@pytest.fixture(scope="module")
def public_surrogate_results():
    matrices, centers = minimax_public_surrogate()
    results = {
        method: solve_minimax(matrices, centers, method=method)
        for method in CANONICAL_METHODS
    }
    return matrices, centers, results


def test_public_surrogate_recipe_matches_literals():
    rng = np.random.default_rng(10602)
    expected_matrices = np.empty((6, 2, 2))
    expected_centers = np.empty((6, 2))
    for instance in range(13):
        matrices = np.empty((6, 2, 2))
        for row in range(6):
            q = np.linalg.qr(rng.standard_normal((2, 2)))[0]
            eigenvalues = np.exp(rng.uniform(-1.0, 1.0, size=2))
            matrices[row] = (q * eigenvalues) @ q.T
        centers = rng.standard_normal((6, 2))

        if instance == 12:
            expected_matrices = matrices
            expected_centers = centers

    matrices, centers = minimax_public_surrogate()
    np.testing.assert_allclose(matrices, expected_matrices, rtol=1e-15, atol=1e-15)
    np.testing.assert_array_equal(centers, expected_centers)


class TestPublicSurrogate:
    def test_slsqp_reference_value_and_gap(self, public_surrogate_results):
        matrices, centers, results = public_surrogate_results
        result = results["scipy-slsqp"]
        gap, residual = _surrogate_diagnostics(result, matrices, centers)

        assert result.converged
        assert abs(result.alpha - SURROGATE_ALPHA) <= 1e-8
        assert gap <= 1e-8
        assert residual <= 1e-8

    @pytest.mark.parametrize(
        "method", [method for method in CANONICAL_METHODS if method != "scipy-slsqp"]
    )
    def test_other_methods_are_accurate_or_report_failure(
        self, method, public_surrogate_results
    ):
        matrices, centers, results = public_surrogate_results
        result = results[method]
        gap, residual = _surrogate_diagnostics(result, matrices, centers)

        if result.converged:
            assert abs(result.alpha - SURROGATE_ALPHA) <= 1e-8
            assert gap <= 1e-8
            assert residual <= 1e-8
        else:
            assert result.converged is False

    def test_newton_failures_do_not_collapse_to_a_vertex(
        self, public_surrogate_results
    ):
        _, _, results = public_surrogate_results
        bisect_error = abs(results["fw+bisect"].alpha - SURROGATE_ALPHA)
        brentq_error = abs(results["fw+brentq"].alpha - SURROGATE_ALPHA)

        for method in ("fw+bisect+newton", "fw+bisect+damped-newton", "newton-cold"):
            assert abs(results[method].alpha - SURROGATE_ALPHA) <= bisect_error
        assert abs(results["fw+brentq+newton"].alpha - SURROGATE_ALPHA) <= brentq_error


# ---------------------------------------------------------------------------
# Basic dispatch / API
# ---------------------------------------------------------------------------


class TestMethodDispatch:
    def test_exactly_seven_canonical_methods(self):
        assert tuple(ALL_METHODS) == CANONICAL_METHODS
        assert "fw+newton" not in ALL_METHODS

    def test_invalid_method_raises(self):
        A = np.eye(2)[np.newaxis]
        x = np.zeros((1, 2))
        with pytest.raises(ValueError, match="Unknown method"):
            solve_minimax(A, x, method="not-a-method")  # type: ignore[arg-type]

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_result_has_method_field(self, method):
        matrices, centers = _random_simplex(3, 2, seed=7)
        res = solve_minimax(matrices, centers, method=method)
        assert res.method == method

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_single_point_trivial(self, method):
        A = np.eye(3)[np.newaxis]
        x = np.array([[1.0, 2.0, 3.0]])
        res = solve_minimax(A, x, method=method)
        assert res.alpha == pytest.approx(0.0, abs=1e-12)
        assert res.converged

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_k0_raises(self, method):
        with pytest.raises(ValueError, match="k=0"):
            solve_minimax(np.zeros((0, 2, 2)), np.zeros((0, 2)), method=method)

    def test_default_method_is_fw_bisect(self):
        matrices, centers = _random_simplex(2, 2)
        res = solve_minimax(matrices, centers)
        assert res.method == "fw+bisect"


# ---------------------------------------------------------------------------
# Numerical agreement across methods
# ---------------------------------------------------------------------------


class TestMethodAgreement:
    """All methods must agree on alpha to within reasonable tolerances."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_pairwise_k2_d2(self, seed):
        """k=2: all methods should agree to 1e-7 with fw+bisect reference."""
        matrices, centers = _random_simplex(2, 2, seed=seed)
        ref = solve_minimax(matrices, centers, method="fw+bisect", tol=1e-12)
        for m in ALL_METHODS:
            if m == "fw+bisect":
                continue
            res = solve_minimax(matrices, centers, method=m)
            assert res.alpha == pytest.approx(
                ref.alpha, rel=1e-7
            ), f"method={m}, seed={seed}: alpha={res.alpha}, ref={ref.alpha}"

    @pytest.mark.parametrize(
        "k,d,seed",
        [
            (3, 2, 10),
            (4, 3, 11),
            (5, 3, 12),
            (3, 5, 13),
        ],
    )
    def test_general_k_d(self, k, d, seed):
        """General k: all methods (except newton-cold) agree to 1e-5."""
        matrices, centers = _random_simplex(k, d, seed=seed)
        ref = solve_minimax(
            matrices, centers, method="fw+bisect", tol=1e-11, max_iter=5000
        )
        for m in ALL_METHODS:
            if m in ("fw+bisect", "newton-cold"):
                continue
            res = solve_minimax(matrices, centers, method=m)
            assert res.alpha == pytest.approx(ref.alpha, rel=1e-5), (
                f"method={m}, k={k}, d={d}, seed={seed}: "
                f"alpha={res.alpha}, ref={ref.alpha}"
            )


# ---------------------------------------------------------------------------
# fw+brentq specific
# ---------------------------------------------------------------------------


class TestFwBrentq:
    @pytest.mark.parametrize("seed", range(5))
    def test_alpha_agrees_with_fw_bisect(self, seed):
        """fw+brentq alpha should match fw+bisect to 1e-7."""
        matrices, centers = _random_simplex(4, 3, seed=seed)
        ref = solve_minimax(matrices, centers, method="fw+bisect", tol=1e-11)
        res = solve_minimax(matrices, centers, method="fw+brentq", tol=1e-11)
        assert res.alpha == pytest.approx(
            ref.alpha, rel=1e-7
        ), f"seed={seed}: brentq={res.alpha}, bisect={ref.alpha}"

    def test_metadata_has_line_search_evals(self):
        matrices, centers = _random_simplex(4, 3, seed=42)
        res = solve_minimax(matrices, centers, method="fw+brentq")
        assert res.metadata is not None
        assert "line_search_evals" in res.metadata
        assert res.metadata["line_search_evals"] > 0


# ---------------------------------------------------------------------------
# fw+brentq+newton specific
# ---------------------------------------------------------------------------


class TestFwBrentqNewton:
    @pytest.mark.parametrize("seed", range(5))
    def test_alpha_precision(self, seed):
        """fw+brentq+newton should achieve high precision."""
        matrices, centers = _random_simplex(4, 3, seed=seed)
        ref = solve_minimax(
            matrices,
            centers,
            method="fw+bisect",
            tol=1e-14,
            max_iter=10000,
        )
        res = solve_minimax(matrices, centers, method="fw+brentq+newton")
        assert res.alpha == pytest.approx(ref.alpha, rel=1e-7)

    def test_metadata_has_newton_info(self):
        matrices, centers = _random_simplex(4, 3, seed=42)
        res = solve_minimax(matrices, centers, method="fw+brentq+newton")
        assert res.metadata is not None
        assert "newton_iters" in res.metadata
        assert "hessian_cond" in res.metadata


# ---------------------------------------------------------------------------
# newton-cold specific
# ---------------------------------------------------------------------------


class TestNewtonCold:
    @pytest.mark.parametrize("seed", range(5))
    def test_pairwise_converges(self, seed):
        """newton-cold should converge for k=2 (pairwise case)."""
        matrices, centers = _random_simplex(2, 2, seed=seed)
        res = solve_minimax(matrices, centers, method="newton-cold")
        ref = solve_minimax(matrices, centers, method="fw+bisect", tol=1e-12)
        assert res.alpha == pytest.approx(
            ref.alpha, rel=1e-4
        ), f"seed={seed}: cold={res.alpha}, ref={ref.alpha}"

    @pytest.mark.parametrize("seed", range(5))
    def test_result_finite(self, seed):
        """newton-cold should produce finite results even if not converged."""
        matrices, centers = _random_simplex(5, 3, seed=seed)
        res = solve_minimax(matrices, centers, method="newton-cold")
        assert np.isfinite(res.alpha)
        assert np.all(np.isfinite(res.circumcenter))


# ---------------------------------------------------------------------------
# fw+bisect+damped-newton specific
# ---------------------------------------------------------------------------


class TestDampedNewton:
    @pytest.mark.parametrize("seed", range(5))
    def test_converges(self, seed):
        matrices, centers = _random_simplex(4, 3, seed=seed)
        res = solve_minimax(matrices, centers, method="fw+bisect+damped-newton")
        ref = solve_minimax(matrices, centers, method="fw+bisect", tol=1e-12)
        assert res.alpha == pytest.approx(ref.alpha, rel=1e-7)

    def test_metadata_has_hessian_cond(self):
        matrices, centers = _random_simplex(4, 3, seed=42)
        res = solve_minimax(matrices, centers, method="fw+bisect+damped-newton")
        assert res.metadata is not None
        assert "hessian_cond" in res.metadata

    def test_empty_thresholded_face_is_nonconverged(self):
        matrices, centers = _random_simplex(3, 2, seed=42)
        res = solve_minimax(
            matrices,
            centers,
            method="fw+bisect+damped-newton",
            weight_tol=1.0,
        )

        assert res.converged is False
        assert res.metadata is not None
        assert res.metadata["newton_status"] == "empty_face"


# ---------------------------------------------------------------------------
# fw+bisect+newton specific
# ---------------------------------------------------------------------------


class TestFwBisectNewton:
    def test_pairwise_higher_precision_than_fw_bisect(self):
        """fw+bisect+newton should achieve tighter residual than fw+bisect alone."""
        matrices, centers = _random_simplex(3, 3, seed=99)

        ref_tight = solve_minimax(
            matrices, centers, method="fw+bisect", tol=1e-14, max_iter=10000
        )
        res_newton = solve_minimax(matrices, centers, method="fw+bisect+newton")

        assert (
            abs(res_newton.alpha - ref_tight.alpha)
            <= abs(
                solve_minimax(matrices, centers, method="fw+bisect").alpha
                - ref_tight.alpha
            )
            + 1e-12
        )

    @pytest.mark.parametrize("seed", range(5))
    def test_converged(self, seed):
        matrices, centers = _random_simplex(4, 3, seed=seed)
        res = solve_minimax(matrices, centers, method="fw+bisect+newton")
        assert res.converged, f"fw+bisect+newton did not converge for seed={seed}"

    def test_active_set_all_equal_fi_at_circumcenter(self):
        """At Newton-polished optimum, all active f_i(x*) should be equal."""
        matrices, centers = _random_simplex(4, 3, seed=42)
        res = solve_minimax(matrices, centers, method="fw+bisect+newton")
        x = res.circumcenter
        active = res.active_set
        f_vals = np.array(
            [
                float(
                    np.einsum("i,ij,j->", x - centers[i], matrices[i], x - centers[i])
                )
                for i in active
            ]
        )
        np.testing.assert_allclose(f_vals, res.alpha, rtol=1e-7, atol=1e-10)


# ---------------------------------------------------------------------------
# Backward compatibility: "fw+newton" alias
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    def test_fw_newton_alias_works(self):
        """'fw+newton' should work as alias for 'fw+bisect+newton'."""
        matrices, centers = _random_simplex(3, 2, seed=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            res = solve_minimax(matrices, centers, method="fw+newton")
        assert res.method == "fw+bisect+newton"

    def test_fw_newton_alias_warns(self):
        matrices, centers = _random_simplex(3, 2, seed=0)
        with pytest.warns(DeprecationWarning, match="deprecated"):
            solve_minimax(matrices, centers, method="fw+newton")


# ---------------------------------------------------------------------------
# scipy-slsqp specific
# ---------------------------------------------------------------------------


class TestScipySlsqp:
    @pytest.mark.parametrize("k", [2, 3, 5])
    def test_alpha_agrees_with_reference(self, k):
        matrices, centers = _random_simplex(k, 3, seed=k * 7)
        ref = solve_minimax(
            matrices, centers, method="fw+bisect", tol=1e-12, max_iter=5000
        )
        res = solve_minimax(matrices, centers, method="scipy-slsqp")
        assert res.alpha == pytest.approx(
            ref.alpha, rel=1e-6
        ), f"scipy-slsqp alpha mismatch for k={k}: {res.alpha} vs ref {ref.alpha}"

    def test_weights_on_simplex(self):
        matrices, centers = _random_simplex(5, 4, seed=55)
        res = solve_minimax(matrices, centers, method="scipy-slsqp")
        assert res.weights.sum() == pytest.approx(1.0, abs=1e-10)
        assert np.all(res.weights >= -1e-12)


# ---------------------------------------------------------------------------
# MinimaxResult backward compatibility
# ---------------------------------------------------------------------------


class TestMinimaxResultBackwardCompat:
    def test_default_method_field(self):
        r = MinimaxResult(
            alpha=1.0,
            circumcenter=np.zeros(2),
            weights=np.array([0.5, 0.5]),
            active_set=[0, 1],
            converged=True,
            n_iter=3,
        )
        assert r.method == "fw+bisect"
        assert r.metadata is None

    def test_namedtuple_positional(self):
        r = MinimaxResult(2.0, np.zeros(2), np.ones(2) * 0.5, [0, 1], True, 5)
        assert r.alpha == 2.0
        assert r.method == "fw+bisect"
        assert r.metadata is None

    def test_metadata_field(self):
        r = MinimaxResult(
            alpha=1.0,
            circumcenter=np.zeros(2),
            weights=np.array([0.5, 0.5]),
            active_set=[0, 1],
            converged=True,
            n_iter=3,
            method="fw+brentq",
            metadata={"fw_iters": 10},
        )
        assert r.metadata == {"fw_iters": 10}
