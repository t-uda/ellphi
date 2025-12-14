import numpy as np
import pytest

from ellphi._solver_python import _u_from_x
from ellphi.solver import pdist_tangency, solve_mu, tangency

from .factories import random_cloud, random_coef_pair


@pytest.fixture
def ellipse_cloud(rng):
    """Return a reproducible ellipse cloud for tangency checks."""
    # 32 ellipses keep the workload representative while staying quick.
    return random_cloud(rng, n_ellipses=32)


def test_pdist_tangency_consistency(ellipse_cloud, solver_backend):
    """Serial and parallel ``pdist_tangency`` implementations agree."""
    serial_result = pdist_tangency(
        ellipse_cloud, parallel=False, backend=solver_backend
    )
    parallel_result = pdist_tangency(
        ellipse_cloud, parallel=True, backend=solver_backend
    )

    np.testing.assert_allclose(
        serial_result,
        parallel_result,
        err_msg="Serial and parallel results are not close enough.",
    )


def test_pdist_tangency_high_dimension(rng):
    cloud = random_cloud(rng, n_ellipses=5, dim=3)
    serial_python = pdist_tangency(cloud, parallel=False, backend="python")
    serial_auto = pdist_tangency(cloud, parallel=False, backend="auto")
    assert serial_python.shape == serial_auto.shape == (5 * 4 // 2,)
    np.testing.assert_allclose(serial_python, serial_auto)


def test_algsig_newton_confines_mu_and_matches_bracket(rng):
    p, q = random_coef_pair(rng, dim=4)
    mu_brent = solve_mu(p, q, method="brentq")
    mu_algsig = solve_mu(p, q, method="algsig+newton", x0=0.5)

    assert 0.0 < mu_algsig < 1.0
    np.testing.assert_allclose(mu_algsig, mu_brent, rtol=1e-9, atol=1e-10)

    algsig_python = tangency(p, q, method="algsig+newton", backend="python", x0=0.5)
    algsig_auto = tangency(p, q, method="algsig+newton", backend="auto", x0=0.5)

    np.testing.assert_allclose(algsig_python.mu, algsig_auto.mu, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(
        algsig_python.point, algsig_auto.point, rtol=1e-9, atol=1e-10
    )


def test_u_from_x_boundaries():
    """_u_from_x handles boundary conditions at 0 and 1 correctly."""
    # At the boundaries, the transform should go to infinity
    assert _u_from_x(0.0) == -np.inf
    assert _u_from_x(1.0) == np.inf

    # Near the boundaries
    eps = 1e-9
    assert np.isfinite(_u_from_x(eps))
    assert _u_from_x(eps) < 0
    assert np.isfinite(_u_from_x(1.0 - eps))
    assert _u_from_x(1.0 - eps) > 0

    # Outside the [0, 1] domain should be NaN
    assert np.isnan(_u_from_x(-0.1))
    assert np.isnan(_u_from_x(1.1))

    # Test with array inputs
    x_array = np.array([0.0, 1.0, 0.5, -0.1, 1.1, 0.25])
    expected = np.array([-np.inf, np.inf, 0.0, np.nan, np.nan, -0.5773502691896257])
    result = _u_from_x(x_array)
    np.testing.assert_allclose(result, expected, equal_nan=True)
