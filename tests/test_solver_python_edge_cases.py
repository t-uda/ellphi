import numpy as np
import pytest
from types import SimpleNamespace

import ellphi._solver_python as solver_py
from ellphi.geometry import coef_from_axes

from .factories import random_cloud, random_coef_pair


def test_gaussian_elimination_requires_square_matrix():
    matrix = np.zeros((2, 3))
    rhs = np.zeros(2)
    with pytest.raises(np.linalg.LinAlgError, match="square"):
        solver_py._gaussian_elimination(matrix, rhs)


def test_gaussian_elimination_detects_singular_pivot():
    matrix = np.array([[0.0, 1.0], [0.0, 2.0]])
    rhs = np.array([1.0, 2.0])
    with pytest.raises(np.linalg.LinAlgError, match="singular"):
        solver_py._gaussian_elimination(matrix, rhs)


def test_gaussian_elimination_detects_zero_diagonal():
    matrix = np.array([[1.0, 1.0], [2.0, 2.0]])
    rhs = np.array([1.0, 2.0])
    with pytest.raises(np.linalg.LinAlgError, match="singular"):
        solver_py._gaussian_elimination(matrix, rhs)


def test_quad_eval_dimension_mismatch():
    coef = coef_from_axes([0.0, 0.0], 1.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="dimensionality"):
        solver_py.quad_eval(coef, [0.0, 0.0, 0.0])


def test_algsig_newton_nan_x0_fallback_converges():
    def f(x: float) -> float:
        return x - 0.5

    def df(x: float) -> float:
        return 1.0

    result = solver_py._algsig_newton_py(f, df, np.nan, maxiter=3, xtol=1e-12, rtol=0.0)
    assert result.converged
    assert result.root == pytest.approx(0.5)


def test_algsig_newton_nonfinite_target_returns_failure():
    def f(x: float) -> float:
        return np.nan

    def df(x: float) -> float:
        return 1.0

    result = solver_py._algsig_newton_py(f, df, 0.5, maxiter=2, xtol=1e-12, rtol=0.0)
    assert not result.converged


def test_algsig_newton_zero_derivative_returns_failure():
    def f(x: float) -> float:
        return 1.0

    def df(x: float) -> float:
        return 0.0

    result = solver_py._algsig_newton_py(f, df, 0.5, maxiter=2, xtol=1e-12, rtol=0.0)
    assert not result.converged


def test_algsig_newton_nonfinite_candidate_backtracks_and_fails():
    def f(x: float) -> float:
        return 1e308

    def df(x: float) -> float:
        return 1e-308

    result = solver_py._algsig_newton_py(f, df, 0.5, maxiter=1, xtol=1e-12, rtol=0.0)
    assert not result.converged


def test_algsig_newton_stops_after_max_iterations():
    def f(x: float) -> float:
        return x - 0.25

    def df(x: float) -> float:
        return 1.0

    result = solver_py._algsig_newton_py(f, df, 0.9, maxiter=1, xtol=0.0, rtol=0.0)
    assert not result.converged


def test_target_prime_returns_nan_when_pencil_build_fails(monkeypatch):
    def raise_error(*args, **kwargs):
        raise np.linalg.LinAlgError("boom")

    monkeypatch.setattr(solver_py, "build_tangent_pencil", raise_error)
    p = coef_from_axes([0.0, 0.0], 1.0, 1.0, 0.0)
    q = coef_from_axes([1.0, 0.0], 1.0, 1.0, 0.0)
    value = solver_py._target_prime(0.5, p, q)
    assert np.isnan(value)


def test_initial_mu_for_newton_defaults_on_all_nan_scores():
    def df(mu: float) -> float:
        return np.nan

    mu = solver_py._initial_mu_for_newton(df)
    assert mu == 0.5


def test_initial_mu_for_newton_prefers_larger_gradient():
    def df(mu: float) -> float:
        return 1.0 if mu < 0.5 else 2.0

    mu = solver_py._initial_mu_for_newton(df)
    assert mu == pytest.approx(1.0 - 1e-5)


def test_solve_mu_brentq_newton_failsafe_uses_brentq(rng, monkeypatch):
    p, q = random_coef_pair(rng)
    expected = solver_py.solve_mu(p, q, method="brentq")

    def raise_newton(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(solver_py, "scipy_newton", raise_newton)
    result = solver_py.solve_mu(p, q, method="brentq+newton", failsafe=True)
    assert result == pytest.approx(expected)


def test_solve_mu_algsig_newton_failsafe_returns_brentq(rng, monkeypatch):
    p, q = random_coef_pair(rng)
    expected = solver_py.solve_mu(p, q, method="brentq")

    monkeypatch.setattr(
        solver_py,
        "_algsig_newton_py",
        lambda *args, **kwargs: solver_py.NewtonResult(0.5, False),
    )

    result = solver_py.solve_mu(p, q, method="algsig+newton", failsafe=True, x0=0.5)
    assert result == pytest.approx(expected)


def test_solve_mu_algsig_newton_without_failsafe_raises(rng, monkeypatch):
    p, q = random_coef_pair(rng)
    monkeypatch.setattr(
        solver_py,
        "_algsig_newton_py",
        lambda *args, **kwargs: solver_py.NewtonResult(0.5, False),
    )

    with pytest.raises(RuntimeError, match="algsig\\+newton failed to converge"):
        solver_py.solve_mu(p, q, method="algsig+newton", failsafe=False, x0=0.5)


def test_solve_mu_newton_nonconverged_failsafe_returns_brentq(rng, monkeypatch):
    p, q = random_coef_pair(rng)
    expected = solver_py.solve_mu(p, q, method="brentq")

    def fake_newton(*args, **kwargs):
        return 0.123, SimpleNamespace(converged=False)

    monkeypatch.setattr(solver_py, "scipy_newton", fake_newton)
    result = solver_py.solve_mu(p, q, method="newton", x0=0.5, failsafe=True)
    assert result == pytest.approx(expected)


def test_solve_mu_newton_nonconverged_without_failsafe_raises(rng, monkeypatch):
    p, q = random_coef_pair(rng)

    def fake_newton(*args, **kwargs):
        return 0.123, SimpleNamespace(converged=False)

    monkeypatch.setattr(solver_py, "scipy_newton", fake_newton)
    with pytest.raises(RuntimeError, match="Newton method failed to converge"):
        solver_py.solve_mu(p, q, method="newton", x0=0.5, failsafe=False)


def test_solve_mu_rejects_nonfinite_target(monkeypatch, rng):
    p, q = random_coef_pair(rng)
    monkeypatch.setattr(solver_py, "_target", lambda mu, p, q: np.nan)
    with pytest.raises(RuntimeError, match="Non-finite target value"):
        solver_py.solve_mu(p, q, method="newton", x0=0.5, failsafe=False)


def test_tangency_rejects_mu_outside_bracket(monkeypatch):
    p = coef_from_axes([0.0, 0.0], 1.0, 1.0, 0.0)
    q = coef_from_axes([1.0, 0.0], 1.0, 1.0, 0.0)
    monkeypatch.setattr(solver_py, "solve_mu", lambda *args, **kwargs: 2.0)
    with pytest.raises(RuntimeError, match="within the bracket"):
        solver_py.tangency(p, q, bracket=(0.0, 1.0))


def test_pdist_tangency_parallel_returns_empty_for_single_ellipse(rng):
    cloud = random_cloud(rng, n_ellipses=1)
    result = solver_py._pdist_tangency_parallel(cloud)
    assert result.shape == (0,)
    assert result.dtype == float


def test_pdist_tangency_serial_branch_matches_serial(rng):
    cloud = random_cloud(rng, n_ellipses=3)
    result = solver_py.pdist_tangency(cloud, parallel=False)
    expected = solver_py._pdist_tangency_serial(cloud)
    np.testing.assert_allclose(result, expected)
