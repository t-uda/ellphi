from typing import Any, cast

import numpy as np
import pytest
from ellphi import coef_from_axes, tangency
from ellphi.solver import quad_eval


# -----------------------------------------------------------------------------
# 1. Unit‑circle tangency (simple, deterministic)
# -----------------------------------------------------------------------------
def test_tangent_unit_circles():
    a = coef_from_axes([0, 0], 1, 1, 0)
    b = coef_from_axes([2, 0], 1, 1, 0)
    res = tangency(a, b)
    assert res.mu == pytest.approx(0.5)
    assert res.point.tolist() == pytest.approx([1.0, 0.0])
    assert res.t == pytest.approx(1.0)


# -----------------------------------------------------------------------------
# 2. Symmetry check with generic, non‑degenerate ellipses
#    (avoid parameters that lead to singular centre computation)
# -----------------------------------------------------------------------------
def test_symmetry_generic():
    p = coef_from_axes([0.3, -0.7], 1.2, 0.9, 0.4)
    q = coef_from_axes([-1.1, 1.4], 0.8, 1.5, 1.0)

    r1 = tangency(p, q)
    r2 = tangency(q, p)

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
def test_newton_requires_x0():
    p = coef_from_axes([0, 0], 1, 1, 0)
    q = coef_from_axes([1, 0], 1, 1, 0)
    with pytest.raises(ValueError):
        tangency(p, q, method="newton")


# -----------------------------------------------------------------------------
# 4. Analytic checks for circles (closed form available)
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "center_p, center_q, radius_p, radius_q",
    [
        ((0.0, 0.0), (2.0, 0.0), 1.0, 1.0),
        ((1.5, -0.5), (5.5, -0.5), 0.5, 1.5),
        ((-1.0, 3.0), (-1.0, -1.0), 2.5, 0.5),
        ((0.0, 0.0), (0.0, 0.0), 1.0, 2.0),
    ],
)
def test_circle_tangency_matches_closed_form(center_p, center_q, radius_p, radius_q):
    center_p = np.asarray(center_p, dtype=float)
    center_q = np.asarray(center_q, dtype=float)
    p = coef_from_axes(center_p, radius_p, radius_p, 0.0)
    q = coef_from_axes(center_q, radius_q, radius_q, 0.0)

    res = tangency(p, q)

    distance = np.linalg.norm(center_p - center_q)
    expected_t = distance / (radius_p + radius_q) if distance else 0.0
    assert res.t == pytest.approx(expected_t, rel=1e-9, abs=1e-12)

    if distance == 0.0:
        assert res.point.tolist() == pytest.approx(center_p.tolist(), rel=1e-9)
    else:
        direction = (center_q - center_p) / distance
        expected_point = center_p + direction * (radius_p * expected_t)
        assert res.point.tolist() == pytest.approx(
            expected_point.tolist(), rel=1e-9, abs=1e-12
        )


# -----------------------------------------------------------------------------
# 5. Axis-aligned ellipses: tangency along the major axis direction
# -----------------------------------------------------------------------------
def test_axis_aligned_ellipses_have_expected_t():
    center_p = np.array([0.0, 0.0])
    center_q = np.array([10.0, 0.0])
    r0_p, r1_p = 3.0, 1.0
    r0_q, r1_q = 1.0, 0.75

    p = coef_from_axes(center_p, r0_p, r1_p, 0.0)
    q = coef_from_axes(center_q, r0_q, r1_q, 0.0)

    res = tangency(p, q)

    expected_t = np.linalg.norm(center_q - center_p) / (r0_p + r0_q)
    assert res.t == pytest.approx(expected_t, rel=1e-12)
    assert res.point.tolist() == pytest.approx([7.5, 0.0], rel=1e-12)
    assert res.mu == pytest.approx(r0_q / (r0_p + r0_q), rel=1e-12)


# -----------------------------------------------------------------------------
# 6. Rotating the entire configuration does not change the tangency scale
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("angle", [0.2, 0.8, 1.3])
def test_rotational_invariance(angle):
    base_center_p = np.array([0.0, 0.0])
    base_center_q = np.array([10.0, 0.0])
    r0_p, r1_p = 3.0, 1.0
    r0_q, r1_q = 1.0, 0.75
    base_t = np.linalg.norm(base_center_q - base_center_p) / (r0_p + r0_q)
    base_point = np.array([base_t * r0_p, 0.0])

    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    center_p = base_center_p @ rot.T
    center_q = base_center_q @ rot.T
    expected_point = base_point @ rot.T

    p = coef_from_axes(center_p, r0_p, r1_p, angle)
    q = coef_from_axes(center_q, r0_q, r1_q, angle)

    res = tangency(p, q)

    assert res.t == pytest.approx(base_t, rel=1e-12)
    assert res.point.tolist() == pytest.approx(expected_point.tolist(), rel=1e-12)
    assert res.mu == pytest.approx(r0_q / (r0_p + r0_q), rel=1e-12)


# -----------------------------------------------------------------------------
# 7. Generic properties at the tangency point
# -----------------------------------------------------------------------------


def _gradient_from_coef(coef: np.ndarray, point: np.ndarray) -> np.ndarray:
    a, b, c, d, e, _ = coef
    x, y = point
    return np.array([2 * a * x + 2 * b * y + 2 * d, 2 * b * x + 2 * c * y + 2 * e])


def test_tangency_point_satisfies_quadratic_and_normal_alignment():
    rng = np.random.default_rng(2024)

    for _ in range(10):
        center_p = rng.uniform(-2.0, 2.0, size=2)

        # Draw a direction vector ensuring sufficient separation between centres
        while True:
            direction = rng.normal(size=2)
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                direction /= norm
                break

        distance = rng.uniform(0.5, 3.0)
        center_q = center_p + direction * distance

        r0_p, r1_p = rng.uniform(0.5, 2.0, size=2)
        r0_q, r1_q = rng.uniform(0.5, 2.0, size=2)
        theta_p, theta_q = rng.uniform(0.0, np.pi, size=2)

        p = coef_from_axes(center_p, r0_p, r1_p, theta_p)
        q = coef_from_axes(center_q, r0_q, r1_q, theta_q)

        res = tangency(p, q)

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
def test_axis_aligned_scaling_matches_sum_of_axes():
    p = coef_from_axes([0.0, 0.0], 0.5, 1.0, 0.0)
    q = coef_from_axes([5.0, 0.0], 2.0, 1.0, 0.0)

    res = tangency(p, q)

    expected_t = 5.0 / (0.5 + 2.0)
    assert res.t == pytest.approx(expected_t, rel=1e-9)

    expected_point = np.array([res.t * 0.5, 0.0])
    assert res.point == pytest.approx(expected_point, abs=1e-12)


# -----------------------------------------------------------------------------
# 9. Tangency point must lie on both scaled ellipses
# -----------------------------------------------------------------------------
def test_tangency_point_satisfies_both_ellipses():
    p = coef_from_axes([0.4, -0.5], 1.1, 0.6, 0.8)
    q = coef_from_axes([-1.3, 0.9], 0.7, 1.4, -0.3)

    res = tangency(p, q)
    expected = res.t**2

    assert res.t >= 0.0
    assert quad_eval(p, res.point) == pytest.approx(expected, rel=1e-9, abs=1e-9)
    assert quad_eval(q, res.point) == pytest.approx(expected, rel=1e-9, abs=1e-9)


# -----------------------------------------------------------------------------
# 10. Translating both ellipses should translate the solution
# -----------------------------------------------------------------------------
def test_tangency_translation_invariance():
    c_p = np.array([-0.2, 0.3])
    c_q = np.array([1.8, -1.1])

    p = coef_from_axes(c_p, 0.9, 0.5, 0.2)
    q = coef_from_axes(c_q, 1.2, 0.7, -0.4)
    base = tangency(p, q)

    shift = np.array([1.1, 0.5])
    p_shift = coef_from_axes(c_p + shift, 0.9, 0.5, 0.2)
    q_shift = coef_from_axes(c_q + shift, 1.2, 0.7, -0.4)
    shifted = tangency(p_shift, q_shift)

    assert shifted.t == pytest.approx(base.t, rel=1e-9)
    assert shifted.mu == pytest.approx(base.mu, rel=1e-9)
    assert shifted.point == pytest.approx(base.point + shift, rel=1e-9, abs=1e-9)


# -----------------------------------------------------------------------------
# 11. Identical ellipses touch immediately (t ≈ 0)
# -----------------------------------------------------------------------------
def test_identical_ellipses_touch_at_zero_time():
    center = np.array([2.3, -1.1])
    ellipse = coef_from_axes(center, 1.0, 0.6, 0.7)

    res = tangency(ellipse, ellipse)

    assert res.t == pytest.approx(0.0, abs=1e-7)
    assert res.point == pytest.approx(center, abs=1e-9)
    assert res.mu == pytest.approx(0.0, abs=1e-12)


# -----------------------------------------------------------------------------
# 12. Alternative scalar-root methods agree with the default strategy
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["bisect", "brentq", "brenth"])
def test_scalar_root_methods_match_default(method: str):
    p = coef_from_axes(cast(Any, [0.3, -0.7]), 1.2, 0.9, 0.4)
    q = coef_from_axes(cast(Any, [-1.1, 1.4]), 0.8, 1.5, 1.0)

    baseline = tangency(p, q)
    alt = tangency(p, q, method=method)

    assert alt.t == pytest.approx(baseline.t, rel=1e-9)
    assert alt.mu == pytest.approx(baseline.mu, rel=1e-9)
    assert alt.point == pytest.approx(baseline.point, rel=1e-9, abs=1e-9)


# -----------------------------------------------------------------------------
# 13. Newton method works when supplied with an initial guess
# -----------------------------------------------------------------------------
def test_newton_method_with_initial_guess():
    p = coef_from_axes([0.3, -0.7], 1.2, 0.9, 0.4)
    q = coef_from_axes([-1.1, 1.4], 0.8, 1.5, 1.0)

    baseline = tangency(p, q)
    res = tangency(p, q, method="newton", x0=0.5)

    assert res.t == pytest.approx(baseline.t, rel=1e-9)
    assert res.mu == pytest.approx(baseline.mu, rel=1e-9)
    assert res.point == pytest.approx(baseline.point, rel=1e-9, abs=1e-9)

    
# -----------------------------------------------------------------------------
# 14. Edge cases for tangency
# -----------------------------------------------------------------------------
def test_tangency_identical_ellipses():
    """Test tangency between two identical ellipses."""
    p = coef_from_axes([0, 0], 2, 1, 0)
    res = tangency(p, p)
    assert res.t == pytest.approx(0.0)


def test_tangency_contained_ellipses():
    """
    Test tangency with one ellipse contained within another.
    The tangency scaling factor `t` should be in the range [0, 1).
    """
    p = coef_from_axes([0, 0], 5, 5, 0)  # Larger ellipse
    q = coef_from_axes([1, 1], 1, 1, 0)  # Smaller ellipse inside
    res = tangency(p, q)
    assert 0 <= res.t < 1


def test_tangency_overlapping_ellipses():
    """
    Test tangency between two overlapping ellipses.
    The tangency scaling factor `t` should be in the range [0, 1).
    """
    p = coef_from_axes([0, 0], 3, 2, 0)
    q = coef_from_axes([1, 0], 3, 2, 0)
    res = tangency(p, q)
    assert 0 <= res.t < 1


def test_tangency_concentric_ellipses():
    """
    Test tangency between two concentric ellipses.
    This test characterizes the current predictable behavior of the algorithm
    for this edge case, which returns t=0 at the center point.
    """
    p = coef_from_axes([0, 0], 2, 1, 0)
    q = coef_from_axes([0, 0], 1, 0.5, 0)  # Smaller, concentric
    res = tangency(p, q)
    assert res.t == pytest.approx(0.0)
    assert res.point.tolist() == pytest.approx([0.0, 0.0])