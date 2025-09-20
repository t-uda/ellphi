import numpy as np
import pytest

from ellphi import coef_from_axes
from ellphi.solver import (
    TangencyResult,
    has_cpp_backend,
    pdist_tangency,
    pdist_tangency_python,
    tangency,
    tangency_python,
)

from .factories import random_cloud


pytestmark = pytest.mark.skipif(
    not has_cpp_backend(), reason="C++ backend is not available"
)


def _assert_results_close(lhs: TangencyResult, rhs: TangencyResult) -> None:
    np.testing.assert_allclose(lhs.t, rhs.t, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(lhs.point, rhs.point, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(lhs.mu, rhs.mu, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "params",
    [
        ((0.0, 0.0), (2.0, 0.0), (1.0, 1.0), (1.0, 1.0), 0.0, 0.0),
        ((0.3, -0.7), (-1.1, 1.4), (1.2, 0.9), (0.8, 1.5), 0.4, 1.0),
        ((-1.5, 0.2), (1.3, -0.4), (0.7, 1.6), (1.1, 0.5), 0.8, 0.3),
    ],
)
def test_tangency_cpp_matches_python(params):
    center_p, center_q, axes_p, axes_q, theta_p, theta_q = params
    p = coef_from_axes(center_p, axes_p[0], axes_p[1], theta_p)
    q = coef_from_axes(center_q, axes_q[0], axes_q[1], theta_q)

    cpp_res = tangency(p, q, backend="cpp")
    py_res = tangency_python(p, q)

    _assert_results_close(cpp_res, py_res)


def test_random_tangencies_match():
    rng = np.random.default_rng(2025)
    for _ in range(10):
        center_p = rng.uniform(-1.0, 1.0, size=2)
        center_q = rng.uniform(-1.0, 1.0, size=2)
        axes_p = rng.uniform(0.5, 2.0, size=2)
        axes_q = rng.uniform(0.5, 2.0, size=2)
        theta_p, theta_q = rng.uniform(0.0, np.pi, size=2)

        p = coef_from_axes(center_p, axes_p[0], axes_p[1], theta_p)
        q = coef_from_axes(center_q, axes_q[0], axes_q[1], theta_q)

        cpp_res = tangency(p, q, backend="cpp")
        py_res = tangency_python(p, q)

        _assert_results_close(cpp_res, py_res)


def test_pdist_cpp_matches_python(rng):
    cloud = random_cloud(rng, n_ellipses=12)

    cpp = pdist_tangency(cloud, backend="cpp")
    py = pdist_tangency_python(cloud)

    np.testing.assert_allclose(cpp, py, rtol=1e-12, atol=1e-12)
