"""Extensive unit tests covering the :mod:`ellphi._tangent_pencil` helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from ellphi import tangency
from ellphi._solver_python import quad_eval
from ellphi._tangent_pencil import (
    TangentPencil,
    build_tangent_pencil,
    center_jacobian,
    linear_vector,
    quad_matrix,
    target_prime_from_pencil,
)
from tests.factories import random_coef_pair


def _circle_coef(center: tuple[float, float], radius: float) -> np.ndarray:
    """Return coefficients for a circle centred at ``center`` with ``radius``."""

    cx, cy = center
    return np.array(
        [1.0, 0.0, 1.0, -cx, -cy, cx * cx + cy * cy - radius * radius],
        dtype=float,
    )


def _gradient(coef: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Return the spatial gradient of a conic at ``point``."""

    a, b, c, d, e, _ = coef
    x, y = point
    return np.array(
        [
            2.0 * a * x + 2.0 * b * y + 2.0 * d,
            2.0 * b * x + 2.0 * c * y + 2.0 * e,
        ]
    )


def _lagrangian_hessian(
    pencil: TangentPencil, p: np.ndarray, q: np.ndarray
) -> np.ndarray:
    """Return the Hessian of ``L(x, μ) = (1-μ) p(x) + μ q(x)`` at the saddle point."""

    grad_diff = _gradient(q, pencil.center) - _gradient(p, pencil.center)
    hessian = np.zeros((3, 3), dtype=float)
    hessian[:2, :2] = 2.0 * pencil.quad
    hessian[:2, 2] = grad_diff
    hessian[2, :2] = grad_diff
    # ``∂²L/∂μ²`` is zero because the Lagrangian is linear in ``μ``.
    return hessian


@dataclass(frozen=True)
class CircleCase:
    """Analytic expectations for circle pairs."""

    center_p: tuple[float, float]
    center_q: tuple[float, float]
    radius_p: float
    radius_q: float

    def coefficients(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            _circle_coef(self.center_p, self.radius_p),
            _circle_coef(self.center_q, self.radius_q),
        )

    @property
    def center_distance(self) -> float:
        return float(np.linalg.norm(np.subtract(self.center_p, self.center_q)))


@pytest.mark.parametrize(
    "case",
    [
        CircleCase((0.0, 0.0), (2.0, 0.0), 1.0, 1.0),
        CircleCase((1.5, -0.5), (5.5, -0.5), 0.5, 1.5),
        CircleCase((-1.0, 3.0), (-1.0, -1.0), 2.5, 0.5),
        CircleCase((0.0, 0.0), (3.5, 0.0), 1.0, 2.0),
    ],
)
def test_tangent_pencil_matches_circle_contact(case: CircleCase) -> None:
    """The cached pencil centre coincides with the tangency point and gradients."""

    p, q = case.coefficients()
    result = tangency(p, q, backend="python")
    pencil = build_tangent_pencil(result.mu, p, q)

    assert pencil.center.tolist() == pytest.approx(
        result.point.tolist(), rel=1e-12, abs=1e-12
    )
    center_tuple: tuple[float, float] = (
        float(pencil.center[0]),
        float(pencil.center[1]),
    )
    assert quad_eval(p, center_tuple) == pytest.approx(
        quad_eval(q, center_tuple), rel=1e-12, abs=1e-12
    )

    grad_p = _gradient(p, pencil.center)
    grad_q = _gradient(q, pencil.center)

    # Stationarity of the Lagrangian: (1-μ)∇p + μ∇q = 0.
    lagrange_stationary = (1.0 - result.mu) * grad_p + result.mu * grad_q
    assert lagrange_stationary.tolist() == pytest.approx(
        [0.0, 0.0], rel=1e-12, abs=1e-12
    )

    # Gradients are colinear at the contact point (cross product vanishes).
    cross = grad_p[0] * grad_q[1] - grad_p[1] * grad_q[0]
    assert cross == pytest.approx(0.0, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(
    "case",
    [
        CircleCase((0.0, 0.0), (2.0, 0.0), 1.0, 1.0),
        CircleCase((1.25, 0.75), (4.25, -0.25), 0.8, 1.4),
        CircleCase((-2.0, 1.0), (2.0, 4.0), 1.5, 0.75),
    ],
)
def test_lagrangian_hessian_is_saddle_for_circles(case: CircleCase) -> None:
    """The joint Lagrangian has an indefinite Hessian at the stationary point."""

    p, q = case.coefficients()
    result = tangency(p, q, backend="python")
    pencil = build_tangent_pencil(result.mu, p, q)
    hessian = _lagrangian_hessian(pencil, p, q)
    eigenvalues = np.linalg.eigvalsh(hessian)

    assert np.min(eigenvalues) < 0.0
    assert np.max(eigenvalues) > 0.0


@pytest.mark.parametrize(
    "case",
    [
        CircleCase((0.0, 0.0), (2.0, 0.0), 1.0, 1.0),
        CircleCase((0.0, 0.0), (3.5, 0.0), 1.25, 0.75),
        CircleCase((1.0, -1.0), (-2.5, 2.0), 0.5, 1.75),
    ],
)
def test_target_prime_matches_circle_closed_form(case: CircleCase) -> None:
    """``∂F/∂μ`` reduces to a simple squared-distance expression for circles."""

    p, q = case.coefficients()
    result = tangency(p, q, backend="python")
    pencil = build_tangent_pencil(result.mu, p, q)
    derivative = target_prime_from_pencil(pencil, p, q)

    expected = 2.0 * case.center_distance**2
    assert derivative == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_target_prime_matches_finite_difference() -> None:
    """The cached derivative matches a numerical differentiation of the target."""

    rng = np.random.default_rng(20240113)
    eps = 1e-6

    for _ in range(5):
        p, q = random_coef_pair(rng)
        result = tangency(p, q, backend="python")
        pencil = build_tangent_pencil(result.mu, p, q)

        def target(mu: float) -> float:
            coef = (1.0 - mu) * p + mu * q
            quad = quad_matrix(coef)
            linear = linear_vector(coef)
            det = quad[0, 0] * quad[1, 1] - quad[0, 1] ** 2
            inv = (1.0 / det) * np.array(
                [[quad[1, 1], -quad[0, 1]], [-quad[0, 1], quad[0, 0]]],
                dtype=float,
            )
            center = -inv @ linear
            center_tuple: tuple[float, float] = (
                float(center[0]),
                float(center[1]),
            )
            return quad_eval(p, center_tuple) - quad_eval(q, center_tuple)

        numerical = (target(result.mu + eps) - target(result.mu - eps)) / (2.0 * eps)
        cached = target_prime_from_pencil(pencil, p, q)
        assert cached == pytest.approx(numerical, rel=5e-7, abs=5e-9)
        assert cached > 0.0


def test_center_jacobian_matches_finite_difference() -> None:
    """``∂x_c/∂r`` agrees with a numerical perturbation of the pencil coefficients."""

    p = _circle_coef((0.5, -1.0), 1.25)
    q = _circle_coef((-2.0, 0.75), 0.9)
    result = tangency(p, q, backend="python")
    pencil = build_tangent_pencil(result.mu, p, q)
    jac = center_jacobian(pencil)

    eps = 1e-6
    for idx in range(6):
        delta = np.zeros_like(pencil.coef)
        delta[idx] = eps
        forward_coef = pencil.coef + delta
        backward_coef = pencil.coef - delta

        def compute_center(coef: np.ndarray) -> np.ndarray:
            quad = quad_matrix(coef)
            linear = linear_vector(coef)
            det = quad[0, 0] * quad[1, 1] - quad[0, 1] ** 2
            inv = (1.0 / det) * np.array(
                [[quad[1, 1], -quad[0, 1]], [-quad[0, 1], quad[0, 0]]],
                dtype=float,
            )
            return -inv @ linear

        fd = (compute_center(forward_coef) - compute_center(backward_coef)) / (
            2.0 * eps
        )
        assert jac[idx].tolist() == pytest.approx(fd.tolist(), rel=2e-6, abs=2e-8)


def test_center_constant_term_has_zero_jacobian() -> None:
    """The constant term of the pencil leaves the centre unchanged."""

    p = _circle_coef((0.0, 0.0), 1.0)
    q = _circle_coef((1.0, 2.0), 0.75)
    result = tangency(p, q, backend="python")
    pencil = build_tangent_pencil(result.mu, p, q)
    jac = center_jacobian(pencil)

    assert jac[5].tolist() == pytest.approx([0.0, 0.0], abs=0.0)
