from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from ellphi.geometry import coef_from_axes
from ellphi._tangent_pencil import (
    build_tangent_pencil,
    center_jacobian,
    linear_vector,
    quad_matrix,
    target_prime_from_pencil,
)
from ellphi.solver import quad_eval, solve_mu, tangency

from .factories import random_coef_pair


class CircleTangencyCase:
    """Analytic circle configuration for TangentPencil validation."""

    def __init__(
        self,
        center_p: np.ndarray,
        center_q: np.ndarray,
        radius_p: float,
        radius_q: float,
    ) -> None:
        self.center_p = np.asarray(center_p, dtype=float)
        self.center_q = np.asarray(center_q, dtype=float)
        self.radius_p = float(radius_p)
        self.radius_q = float(radius_q)

    def coefficients(self) -> tuple[np.ndarray, np.ndarray]:
        """Return coefficient vectors for the stored circles."""

        return (
            coef_from_axes(cast(Any, self.center_p), self.radius_p, self.radius_p, 0.0),
            coef_from_axes(cast(Any, self.center_q), self.radius_q, self.radius_q, 0.0),
        )

    @property
    def distance(self) -> float:
        """Center distance between the circles."""

        return float(np.linalg.norm(self.center_q - self.center_p))

    @property
    def mu(self) -> float:
        """Analytic μ associated with external tangency."""

        return self.radius_q / (self.radius_p + self.radius_q)

    @property
    def contact_point(self) -> np.ndarray:
        """Closed-form tangency point between the two circles."""

        diff = self.center_q - self.center_p
        distance = self.distance
        direction = diff / distance
        offset = self.radius_p * distance / (self.radius_p + self.radius_q)
        return self.center_p + direction * offset

    def analytic_target_prime(self) -> float:
        """Return analytic derivative of the tangency constraint."""

        mu = self.mu
        rp2 = self.radius_p**2
        rq2 = self.radius_q**2
        denom = (1.0 - mu) * rq2 + mu * rp2
        return 2.0 * (self.distance**2) * rp2 * rq2 / (denom**3)


CIRCLE_CASES = [
    CircleTangencyCase(np.array([0.0, 0.0]), np.array([2.0, 0.0]), 1.0, 1.0),
    CircleTangencyCase(np.array([1.5, -0.5]), np.array([5.5, -0.5]), 0.5, 1.5),
    CircleTangencyCase(np.array([-1.0, 3.0]), np.array([-1.0, -1.5]), 2.5, 0.75),
    CircleTangencyCase(np.array([2.0, 3.0]), np.array([-4.0, -1.0]), 1.5, 0.75),
    CircleTangencyCase(np.array([3.0, -2.0]), np.array([0.5, 4.5]), 2.0, 1.25),
]


def _as_point(center: np.ndarray) -> tuple[float, float]:
    """Return the center as a tuple compatible with `quad_eval`."""

    return (float(center[0]), float(center[1]))


@pytest.mark.parametrize("case", CIRCLE_CASES)
def test_circle_center_and_gradients_match_hand_solution(case: CircleTangencyCase):
    """Circle cases exercise the full Lagrange conditions analytically."""

    p, q = case.coefficients()
    mu = case.mu
    pencil = build_tangent_pencil(mu, p, q)

    np.testing.assert_allclose(
        pencil.coef, (1.0 - mu) * p + mu * q, rtol=1e-14, atol=1e-14
    )

    np.testing.assert_allclose(
        pencil.center, case.contact_point, rtol=1e-12, atol=1e-12
    )

    point = _as_point(pencil.center)
    val_p = quad_eval(p, point)
    val_q = quad_eval(q, point)
    assert val_p == pytest.approx(val_q, rel=1e-12, abs=1e-12)

    grad_combo = pencil.quad @ pencil.center + pencil.linear
    np.testing.assert_allclose(grad_combo, np.zeros(2), atol=1e-12)

    grad_p = 2.0 * (quad_matrix(p) @ pencil.center + linear_vector(p))
    grad_q = 2.0 * (quad_matrix(q) @ pencil.center + linear_vector(q))

    residual = (1.0 - mu) * grad_p + mu * grad_q
    np.testing.assert_allclose(residual, np.zeros_like(residual), atol=1e-12)

    cross = grad_p[0] * grad_q[1] - grad_p[1] * grad_q[0]
    assert cross == pytest.approx(0.0, abs=1e-12)

    solver_point = tangency(p, q).point
    np.testing.assert_allclose(solver_point, pencil.center, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("case", CIRCLE_CASES)
def test_solver_mu_matches_circle_formula(case: CircleTangencyCase):
    """`solve_mu` reproduces the closed-form μ for analytic circles."""

    p, q = case.coefficients()
    mu_numeric = solve_mu(p, q)
    assert mu_numeric == pytest.approx(case.mu, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("case", CIRCLE_CASES)
def test_circle_target_prime_matches_closed_form(case: CircleTangencyCase):
    """`target_prime_from_pencil` agrees with the analytic derivative."""

    p, q = case.coefficients()
    mu = case.mu
    pencil = build_tangent_pencil(mu, p, q)

    expected = case.analytic_target_prime()
    result = target_prime_from_pencil(pencil, p, q)
    assert result == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("case", CIRCLE_CASES)
def test_circle_center_jacobian_matches_manual_formula(case: CircleTangencyCase):
    """The symbolic Jacobian equals the hand-derived expression for circles."""

    p, q = case.coefficients()
    mu = case.mu
    pencil = build_tangent_pencil(mu, p, q)
    jac = center_jacobian(pencil)

    a = pencil.quad[0, 0]
    xc, yc = pencil.center
    expected = np.zeros_like(jac)
    expected[0] = np.array([-xc / a, 0.0])
    expected[1] = np.array([-yc / a, -xc / a])
    expected[2] = np.array([0.0, -yc / a])
    expected[3] = np.array([-1.0 / a, 0.0])
    expected[4] = np.array([0.0, -1.0 / a])

    np.testing.assert_allclose(jac, expected, rtol=1e-12, atol=1e-12)


def test_center_jacobian_matches_finite_difference(rng: np.random.Generator):
    """`center_jacobian` matches a central-difference approximation."""

    for _ in range(5):
        p, q = random_coef_pair(rng)
        mu = solve_mu(p, q)
        pencil = build_tangent_pencil(mu, p, q)
        jac = center_jacobian(pencil)

        for idx in range(6):
            step = 1e-6
            coef_plus = pencil.coef.copy()
            coef_minus = pencil.coef.copy()
            coef_plus[idx] += step
            coef_minus[idx] -= step

            quad_plus = quad_matrix(coef_plus)
            quad_minus = quad_matrix(coef_minus)
            linear_plus = linear_vector(coef_plus)
            linear_minus = linear_vector(coef_minus)

            center_plus = -np.linalg.solve(quad_plus, linear_plus)
            center_minus = -np.linalg.solve(quad_minus, linear_minus)
            finite_diff = (center_plus - center_minus) / (2.0 * step)

            np.testing.assert_allclose(finite_diff, jac[idx], rtol=1e-8, atol=1e-8)


def _target_value(mu: float, p: np.ndarray, q: np.ndarray) -> float:
    pencil = build_tangent_pencil(mu, p, q)
    center = _as_point(pencil.center)
    return float(quad_eval(p, center) - quad_eval(q, center))


def test_target_prime_chain_rule_matches_closed_form(rng: np.random.Generator):
    """Chain rule using `center_jacobian` reproduces the cached derivative."""

    for _ in range(5):
        p, q = random_coef_pair(rng)
        mu = solve_mu(p, q)
        pencil = build_tangent_pencil(mu, p, q)
        jac = center_jacobian(pencil)

        diff_coef = q - p
        center_prime = diff_coef @ jac

        diff = p - q
        grad_diff = 2.0 * (quad_matrix(diff) @ pencil.center + linear_vector(diff))
        derivative_chain = float(grad_diff @ center_prime)

        derivative_cached = target_prime_from_pencil(pencil, p, q)
        assert derivative_chain == pytest.approx(
            derivative_cached, rel=1e-10, abs=1e-10
        )


def test_target_prime_matches_finite_difference(rng: np.random.Generator):
    """`target_prime_from_pencil` matches a finite-difference baseline."""

    for _ in range(5):
        p, q = random_coef_pair(rng)
        mu = solve_mu(p, q)
        pencil = build_tangent_pencil(mu, p, q)

        step = min(1e-6, mu, 1.0 - mu) * 0.5
        if step == 0.0:
            step = 1e-8

        deriv_fd = (_target_value(mu + step, p, q) - _target_value(mu - step, p, q)) / (
            2.0 * step
        )
        deriv_exact = target_prime_from_pencil(pencil, p, q)

        assert deriv_exact == pytest.approx(deriv_fd, rel=1e-8, abs=1e-8)


def test_lagrange_conditions_hold_for_random_pairs(rng: np.random.Generator):
    """Random ellipses satisfy the Lagrange multiplier conditions."""

    for _ in range(5):
        p, q = random_coef_pair(rng)
        mu = solve_mu(p, q)
        pencil = build_tangent_pencil(mu, p, q)
        center = pencil.center

        grad_combo = pencil.quad @ center + pencil.linear
        np.testing.assert_allclose(grad_combo, np.zeros(2), atol=1e-10)

        center_point = _as_point(center)
        val_diff = quad_eval(p, center_point) - quad_eval(q, center_point)
        assert val_diff == pytest.approx(0.0, abs=1e-10)

        grad_p = 2.0 * (quad_matrix(p) @ center + linear_vector(p))
        grad_q = 2.0 * (quad_matrix(q) @ center + linear_vector(q))

        residual = (1.0 - mu) * grad_p + mu * grad_q
        np.testing.assert_allclose(residual, np.zeros_like(residual), atol=1e-10)

        cross = grad_p[0] * grad_q[1] - grad_p[1] * grad_q[0]
        assert cross == pytest.approx(0.0, abs=1e-10)


def test_saddle_point_behaviour(rng: np.random.Generator):
    """The Lagrangian exhibits saddle behaviour at the tangency solution."""

    for _ in range(5):
        p, q = random_coef_pair(rng)
        mu = solve_mu(p, q)
        pencil = build_tangent_pencil(mu, p, q)

        eigenvalues = np.linalg.eigvalsh(pencil.quad)
        assert np.all(eigenvalues > 0.0)

        step = min(1e-3, mu, 1.0 - mu) * 0.5
        if step == 0.0:
            step = 1e-6

        t_minus = _target_value(mu - step, p, q)
        t_plus = _target_value(mu + step, p, q)
        assert t_minus < 0.0 < t_plus

        derivative = target_prime_from_pencil(pencil, p, q)
        assert derivative > 0.0


def test_target_prime_vanishes_for_identical_conics() -> None:
    """Derivative of the tangency target is zero when the conics coincide."""

    coef = coef_from_axes(cast(Any, np.array([0.25, -0.75])), 1.5, 0.5, 0.3)
    mu = 0.37
    pencil = build_tangent_pencil(mu, coef, coef)
    derivative = target_prime_from_pencil(pencil, coef, coef)
    assert derivative == pytest.approx(0.0, abs=1e-15)


def test_build_tangent_pencil_raises_on_singular_quadratic() -> None:
    """`build_tangent_pencil` fails when the quadratic form is singular."""

    p = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    q = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    with pytest.raises(ZeroDivisionError):
        build_tangent_pencil(0.0, p, q)
