import numpy as np
import pytest
from scipy import linalg

from ellphi._solver_python import _center, _gaussian_elimination
from ellphi.geometry import pack_conic
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


def test_python_center_uses_gaussian_fallback_when_cholesky_fails():
    A = np.array([[0.0, 2.0], [2.0, 3.0]], dtype=float)
    b = np.array([1.0, -4.0], dtype=float)
    coef = pack_conic(A, b, 0.0)

    with pytest.raises(linalg.LinAlgError):
        linalg.cho_factor(A, check_finite=False)

    expected = np.array([2.75, -0.5])

    center = _center(coef)
    np.testing.assert_allclose(center, expected)
    np.testing.assert_allclose(_gaussian_elimination(A, -b), expected)
