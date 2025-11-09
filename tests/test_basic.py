from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pytest
from ellphi import coef_from_axes, coef_from_cov, tangency
from ellphi.geometry import unpack_single_conic
from ellphi.solver import quad_eval, tangency as solver_tangency
from tests.factories import random_coef_pair, rotation_matrix


@dataclass(frozen=True)
class CircleCase:
    """Represent a pair of circles with analytic tangency expectations."""

    center_p: np.ndarray
    center_q: np.ndarray
    radius_p: float
    radius_q: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "center_p", np.asarray(self.center_p, dtype=float))
        object.__setattr__(self, "center_q", np.asarray(self.center_q, dtype=float))

    def coefficients(self) -> tuple[np.ndarray, np.ndarray]:
        """Return quadratic coefficients for the stored circles."""
        return (
            coef_from_axes(cast(Any, self.center_p), self.radius_p, self.radius_p, 0.0),
            coef_from_axes(cast(Any, self.center_q), self.radius_q, self.radius_q, 0.0),
        )

    @property
    def distance(self) -> float:
        return float(np.linalg.norm(self.center_p - self.center_q))

    @property
    def expected_t(self) -> float:
        distance = self.distance
        if distance == 0.0:
            return 0.0
        return distance / (self.radius_p + self.radius_q)

    @property
    def expected_point(self) -> np.ndarray:
        distance = self.distance
        if distance == 0.0:
            return self.center_p
        direction = (self.center_q - self.center_p) / distance
        return self.center_p + direction * (self.radius_p * self.expected_t)


@dataclass(frozen=True)
class AxisAlignedCase:
    """Represent an axis-aligned ellipse pair used to probe invariance properties."""

    center_p: np.ndarray
    center_q: np.ndarray
    axes_p: tuple[float, float]
    axes_q: tuple[float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "center_p", np.asarray(self.center_p, dtype=float))
        object.__setattr__(self, "center_q", np.asarray(self.center_q, dtype=float))

    @property
    def distance(self) -> float:
        return float(np.linalg.norm(self.center_q - self.center_p))

    @property
    def expected_t(self) -> float:
        return self.distance / (self.axes_p[0] + self.axes_q[0])

    @property
    def expected_mu(self) -> float:
        return self.axes_q[0] / (self.axes_p[0] + self.axes_q[0])

    @property
    def expected_point(self) -> np.ndarray:
        if self.distance == 0.0:
            return self.center_p
        direction = (self.center_q - self.center_p) / self.distance
        return self.center_p + direction * (self.axes_p[0] * self.expected_t)

    def coefficients(self) -> tuple[np.ndarray, np.ndarray]:
        """Return coefficients for the default axis-aligned configuration."""
        return self.coefficients_with_orientation()

    def coefficients_with_orientation(
        self,
        *,
        angle_p: float = 0.0,
        angle_q: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return coefficients after applying optional rotations."""
        r0_p, r1_p = self.axes_p
        r0_q, r1_q = self.axes_q
        return (
            coef_from_axes(cast(Any, self.center_p), r0_p, r1_p, angle_p),
            coef_from_axes(cast(Any, self.center_q), r0_q, r1_q, angle_q),
        )


@pytest.fixture
def axis_aligned_case() -> AxisAlignedCase:
    return AxisAlignedCase(
        center_p=np.array([0.0, 0.0], dtype=float),
        center_q=np.array([10.0, 0.0], dtype=float),
        axes_p=(3.0, 1.0),
        axes_q=(1.0, 0.75),
    )


# -----------------------------------------------------------------------------
# 1. Unit‑circle tangency (simple, deterministic)
# -----------------------------------------------------------------------------
def test_tangent_unit_circles(solver_backend):
    a = coef_from_axes([0, 0], 1, 1, 0)
    b = coef_from_axes([2, 0], 1, 1, 0)
    res = tangency(a, b, backend=solver_backend)
    assert res.mu == pytest.approx(0.5)
    assert res.point.tolist() == pytest.approx([1.0, 0.0])
    assert res.t == pytest.approx(1.0)


def test_tangency_three_dimensional_spheres(solver_backend):
    center_p = np.array([0.0, 0.0, 0.0], dtype=float)
    center_q = np.array([3.0, 0.0, 0.0], dtype=float)
    cov = np.eye(3)
    p = coef_from_cov(center_p, cov)
    q = coef_from_cov(center_q, cov)
    res = tangency(p, q, backend=solver_backend)
    assert res.mu == pytest.approx(0.5)
    np.testing.assert_allclose(res.point, np.array([1.5, 0.0, 0.0]))
    assert res.t == pytest.approx(1.5)


@pytest.mark.parametrize("dim", [3, 4])
def test_tangency_random_high_dimension(
    dim: int, rng: np.random.Generator, solver_backend: str
) -> None:
    iterations = 5
    for _ in range(iterations):
        pcoef, qcoef = random_coef_pair(rng, dim=dim)
        result = solver_tangency(pcoef, qcoef, backend=solver_backend)

        point = result.point
        assert point.shape == (dim,)

        p_value = quad_eval(pcoef, point)
        q_value = quad_eval(qcoef, point)
        expected = result.t**2
        assert p_value == pytest.approx(expected, rel=1e-7, abs=1e-10)
        assert q_value == pytest.approx(expected, rel=1e-7, abs=1e-10)

        Ap, bp, _ = unpack_single_conic(pcoef)
        Aq, bq, _ = unpack_single_conic(qcoef)
        grad_p = 2.0 * (Ap @ point) + 2.0 * bp
        grad_q = 2.0 * (Aq @ point) + 2.0 * bq

        combination = (1.0 - result.mu) * grad_p + result.mu * grad_q
        np.testing.assert_allclose(
            combination,
            np.zeros_like(point),
            rtol=1e-6,
            atol=1e-6,
        )


# -----------------------------------------------------------------------------
# 2. Symmetry check with generic, non‑degenerate ellipses
#    (avoid parameters that lead to singular centre computation)
# -----------------------------------------------------------------------------
def test_symmetry_generic(solver_backend):
    p = coef_from_axes([0.3, -0.7], 1.2, 0.9, 0.4)
    q = coef_from_axes([-1.1, 1.4], 0.8, 1.5, 1.0)

    r1 = tangency(p, q, backend=solver_backend)
    r2 = tangency(q, p, backend=solver_backend)

    # Ensure results are finite
    for r in (r1, r2):
        assert not np.isnan(r.t)
        assert not np.isnan(r.mu)
        assert np.all(np.isfinite(r.point))

    # Distances and points should match; mu should complement
    assert r1.t == pytest.approx(r2.t, rel=1e-6)
    assert r1.point.tolist() == pytest.approx(r2.point.tolist(), rel=1e-6)
    assert r1.mu == pytest.approx(1.0 - r2.mu, rel=1e-6)


# -----------------------------------------------------------------------------
# 3. Error handling: Newton method requires x0
# -----------------------------------------------------------------------------
def test_newton_requires_x0(solver_backend):
    p = coef_from_axes([0, 0], 1, 1, 0)
    q = coef_from_axes([1, 0], 1, 1, 0)
    with pytest.raises(ValueError):
        tangency(p, q, method="newton", backend=solver_backend)


# -----------------------------------------------------------------------------
# 4. Analytic checks for circles (closed form available)
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "case",
    [
        CircleCase(
            np.array([0.0, 0.0], dtype=float),
            np.array([2.0, 0.0], dtype=float),
            1.0,
            1.0,
        ),
        CircleCase(
            np.array([1.5, -0.5], dtype=float),
            np.array([5.5, -0.5], dtype=float),
            0.5,
            1.5,
        ),
        CircleCase(
            np.array([-1.0, 3.0], dtype=float),
            np.array([-1.0, -1.0], dtype=float),
            2.5,
            0.5,
        ),
        CircleCase(
            np.array([0.0, 0.0], dtype=float),
            np.array([0.0, 0.0], dtype=float),
            1.0,
            2.0,
        ),
    ],
)
def test_circle_tangency_matches_closed_form(case: CircleCase, solver_backend):
    p, q = case.coefficients()
    res = tangency(p, q, backend=solver_backend)

    assert res.t == pytest.approx(case.expected_t, rel=1e-9, abs=1e-12)
    expected_point = case.expected_point
    assert res.point.tolist() == pytest.approx(
        expected_point.tolist(), rel=1e-9, abs=1e-12
    )


# -----------------------------------------------------------------------------
# 5. Axis-aligned ellipses: tangency along the major axis direction
# -----------------------------------------------------------------------------
def test_axis_aligned_ellipses_have_expected_t(
    axis_aligned_case: AxisAlignedCase, solver_backend
):
    p, q = axis_aligned_case.coefficients()
    res = tangency(p, q, backend=solver_backend)

    assert res.t == pytest.approx(axis_aligned_case.expected_t, rel=1e-12)
    assert res.point.tolist() == pytest.approx(
        axis_aligned_case.expected_point.tolist(), rel=1e-12
    )
    assert res.mu == pytest.approx(axis_aligned_case.expected_mu, rel=1e-12)


# -----------------------------------------------------------------------------
# 6. Rotating the entire configuration does not change the tangency scale
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("angle", [0.2, 0.8, 1.3])
def test_rotational_invariance(
    angle: float, axis_aligned_case: AxisAlignedCase, solver_backend
):
    rot = rotation_matrix(angle)
    center_p = axis_aligned_case.center_p @ rot.T
    center_q = axis_aligned_case.center_q @ rot.T
    expected_point = axis_aligned_case.expected_point @ rot.T

    r0_p, r1_p = axis_aligned_case.axes_p
    r0_q, r1_q = axis_aligned_case.axes_q
    p = coef_from_axes(cast(Any, center_p), r0_p, r1_p, angle)
    q = coef_from_axes(cast(Any, center_q), r0_q, r1_q, angle)

    res = tangency(p, q, backend=solver_backend)

    assert res.t == pytest.approx(axis_aligned_case.expected_t, rel=1e-12)
    assert res.point.tolist() == pytest.approx(expected_point.tolist(), rel=1e-12)
    assert res.mu == pytest.approx(axis_aligned_case.expected_mu, rel=1e-12)


# -----------------------------------------------------------------------------
# 7. Generic properties at the tangency point
# -----------------------------------------------------------------------------


def _gradient_from_coef(coef: np.ndarray, point: np.ndarray) -> np.ndarray:
    a, b, c, d, e, _ = coef
    x, y = point
    return np.array([2 * a * x + 2 * b * y + 2 * d, 2 * b * x + 2 * c * y + 2 * e])


def test_tangency_point_satisfies_quadratic_and_normal_alignment(solver_backend):
    rng = np.random.default_rng(2024)

    for _ in range(10):
        p, q = random_coef_pair(rng)

        res = tangency(p, q, backend=solver_backend)

        assert res.t >= 0.0
        assert -1e-9 <= res.mu <= 1.0 + 1e-9

        value_p = quad_eval(p, res.point)
        value_q = quad_eval(q, res.point)
        assert value_p == pytest.approx(res.t**2, rel=1e-9, abs=1e-9)
        assert value_q == pytest.approx(res.t**2, rel=1e-9, abs=1e-9)

        grad_p = _gradient_from_coef(p, res.point)
        grad_q = _gradient_from_coef(q, res.point)
        cross = grad_p[0] * grad_q[1] - grad_p[1] * grad_q[0]
        assert cross == pytest.approx(0.0, abs=1e-8)
        assert np.dot(grad_p, grad_q) <= 1e-8


# -----------------------------------------------------------------------------
# 8. Axis-aligned ellipses: tangency time matches analytic distance formula
# -----------------------------------------------------------------------------
def test_axis_aligned_scaling_matches_sum_of_axes(solver_backend):
    case = AxisAlignedCase(
        center_p=np.array([0.0, 0.0], dtype=float),
        center_q=np.array([5.0, 0.0], dtype=float),
        axes_p=(0.5, 1.0),
        axes_q=(2.0, 1.0),
    )
    p, q = case.coefficients()

    res = tangency(p, q, backend=solver_backend)

    assert res.t == pytest.approx(case.expected_t, rel=1e-9)
    assert res.point.tolist() == pytest.approx(case.expected_point.tolist(), abs=1e-12)


# -----------------------------------------------------------------------------
# 9. Tangency point must lie on both scaled ellipses
# -----------------------------------------------------------------------------
def test_tangency_point_satisfies_both_ellipses(solver_backend):
    p = coef_from_axes([0.4, -0.5], 1.1, 0.6, 0.8)
    q = coef_from_axes([-1.3, 0.9], 0.7, 1.4, -0.3)

    res = tangency(p, q, backend=solver_backend)
    expected = res.t**2

    assert res.t >= 0.0
    assert quad_eval(p, res.point) == pytest.approx(expected, rel=1e-9, abs=1e-9)
    assert quad_eval(q, res.point) == pytest.approx(expected, rel=1e-9, abs=1e-9)


# -----------------------------------------------------------------------------
# 10. Translating both ellipses should translate the solution
# -----------------------------------------------------------------------------
def test_tangency_translation_invariance(solver_backend):
    c_p = np.array([-0.2, 0.3])
    c_q = np.array([1.8, -1.1])

    p = coef_from_axes(c_p, 0.9, 0.5, 0.2)
    q = coef_from_axes(c_q, 1.2, 0.7, -0.4)
    base = tangency(p, q, backend=solver_backend)

    shift = np.array([1.1, 0.5])
    p_shift = coef_from_axes(c_p + shift, 0.9, 0.5, 0.2)
    q_shift = coef_from_axes(c_q + shift, 1.2, 0.7, -0.4)
    shifted = tangency(p_shift, q_shift, backend=solver_backend)

    assert shifted.t == pytest.approx(base.t, rel=1e-9)
    assert shifted.mu == pytest.approx(base.mu, rel=1e-9)
    assert shifted.point == pytest.approx(base.point + shift, rel=1e-9, abs=1e-9)


# -----------------------------------------------------------------------------
# 11. Identical ellipses touch immediately (t ≈ 0)
# -----------------------------------------------------------------------------
def test_identical_ellipses_touch_at_zero_time(solver_backend):
    center = np.array([2.3, -1.1])
    ellipse = coef_from_axes(center, 1.0, 0.6, 0.7)

    res = tangency(ellipse, ellipse, backend=solver_backend)

    assert res.t == pytest.approx(0.0, abs=1e-7)
    assert res.point == pytest.approx(center, abs=1e-9)
    assert res.mu == pytest.approx(0.0, abs=1e-12)


# -----------------------------------------------------------------------------
# 12. Alternative scalar-root methods agree with the default strategy
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["bisect", "brentq", "brenth"])
def test_scalar_root_methods_match_default(method: str, solver_backend):
    p = coef_from_axes(cast(Any, [0.3, -0.7]), 1.2, 0.9, 0.4)
    q = coef_from_axes(cast(Any, [-1.1, 1.4]), 0.8, 1.5, 1.0)

    baseline = tangency(p, q, backend=solver_backend)
    alt = tangency(p, q, method=method, backend=solver_backend)

    assert alt.t == pytest.approx(baseline.t, rel=1e-9)
    assert alt.mu == pytest.approx(baseline.mu, rel=1e-9)
    assert alt.point == pytest.approx(baseline.point, rel=1e-9, abs=1e-9)


# -----------------------------------------------------------------------------
# 13. Newton method works when supplied with an initial guess
# -----------------------------------------------------------------------------
def test_newton_method_with_initial_guess(solver_backend):
    p = coef_from_axes([0.3, -0.7], 1.2, 0.9, 0.4)
    q = coef_from_axes([-1.1, 1.4], 0.8, 1.5, 1.0)

    baseline = tangency(p, q, backend=solver_backend)
    res = tangency(p, q, method="newton", x0=0.5, backend=solver_backend)

    assert res.t == pytest.approx(baseline.t, rel=1e-9)
    assert res.mu == pytest.approx(baseline.mu, rel=1e-9)
    assert res.point == pytest.approx(baseline.point, rel=1e-9, abs=1e-9)


# -----------------------------------------------------------------------------
# 14. Edge cases for tangency
# -----------------------------------------------------------------------------
def test_tangency_identical_ellipses(solver_backend):
    """Test tangency between two identical ellipses."""
    p = coef_from_axes([0, 0], 2, 1, 0)
    res = tangency(p, p, backend=solver_backend)
    assert res.t == pytest.approx(0.0)


def test_tangency_contained_ellipses(solver_backend):
    """
    Test tangency with one ellipse contained within another.
    The tangency scaling factor `t` should be in the range [0, 1).
    """
    p = coef_from_axes([0, 0], 5, 5, 0)  # Larger ellipse
    q = coef_from_axes([1, 1], 1, 1, 0)  # Smaller ellipse inside
    res = tangency(p, q, backend=solver_backend)
    assert 0 <= res.t < 1


def test_tangency_overlapping_ellipses(solver_backend):
    """
    Test tangency between two overlapping ellipses.
    The tangency scaling factor `t` should be in the range [0, 1).
    """
    p = coef_from_axes([0, 0], 3, 2, 0)
    q = coef_from_axes([1, 0], 3, 2, 0)
    res = tangency(p, q, backend=solver_backend)
    assert 0 <= res.t < 1


def test_tangency_concentric_ellipses(solver_backend):
    """
    Test tangency between two concentric ellipses.
    This test characterizes the current predictable behavior of the algorithm
    for this edge case, which returns t=0 at the center point.
    """
    p = coef_from_axes([0, 0], 2, 1, 0)
    q = coef_from_axes([0, 0], 1, 0.5, 0)  # Smaller, concentric
    res = tangency(p, q, backend=solver_backend)
    assert res.t == pytest.approx(0.0)
    assert res.point.tolist() == pytest.approx([0.0, 0.0])
