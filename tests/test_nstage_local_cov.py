import numpy as np
import pytest
from ellphi.ellcloud import ellipse_cloud, EllipseCloud


def test_nstage_local_cov_smoke():
    """Smoke test for the n-stage local covariance method."""
    rng = np.random.default_rng(42)
    X = rng.uniform(size=(20, 2))
    k = 4
    n_stages = 2

    # Run the n-stage local covariance
    ellcloud = ellipse_cloud(
        X,
        method="nstage_local_cov",
        k=k,
        n_stages=n_stages,
    )

    # Check that the output is an EllipseCloud instance
    assert isinstance(ellcloud, EllipseCloud)

    # Check that the number of ellipses is correct
    assert ellcloud.n <= X.shape[0]
    assert ellcloud.k == k


def test_nstage_local_cov_invalid_stages():
    """Test that n_stages < 1 raises a ValueError."""
    rng = np.random.default_rng(42)
    X = rng.uniform(size=(20, 2))
    with pytest.raises(ValueError):
        ellipse_cloud(X, method="nstage_local_cov", k=4, n_stages=0)
    with pytest.raises(ValueError):
        ellipse_cloud(X, method="nstage_local_cov", k=4, n_stages=-1)


def test_nstage_one_stage_is_equivalent_to_local_cov():
    """Test that n_stages=1 gives the same result as the standard local_cov."""
    rng = np.random.default_rng(42)
    X = rng.uniform(size=(20, 2))
    k = 4

    # Run the 1-stage n-stage local covariance
    ellcloud_nstage = ellipse_cloud(
        X, method="nstage_local_cov", k=k, n_stages=1
    )

    # Run the standard local covariance
    ellcloud_local_cov = ellipse_cloud(X, method="local_cov", k=k)

    # Check that the results are identical
    np.testing.assert_array_almost_equal(
        ellcloud_nstage.coef, ellcloud_local_cov.coef
    )
    np.testing.assert_array_almost_equal(
        ellcloud_nstage.mean, ellcloud_local_cov.mean
    )
    np.testing.assert_array_almost_equal(
        ellcloud_nstage.cov, ellcloud_local_cov.cov
    )
    assert ellcloud_nstage.k == ellcloud_local_cov.k
    np.testing.assert_array_equal(ellcloud_nstage.nbd, ellcloud_local_cov.nbd)
