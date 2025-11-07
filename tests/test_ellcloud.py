import numpy as np
import pytest

from ellphi.ellcloud import EllipseCloud
from ellphi.geometry import coef_from_cov


def test_local_cov_uses_actual_neighbourhood_size():
    # Request more neighbours than available to trigger neighbourhood truncation.
    X = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.5]])

    ellcloud = EllipseCloud.from_local_cov(X, k=5)

    # Only one unique neighbourhood should exist and it should contain all points.
    assert ellcloud.cov.shape[0] == 1
    expected_cov = np.cov(X, rowvar=False, bias=False)
    assert ellcloud.cov[0] == pytest.approx(expected_cov)


def test_local_cov_requires_k_at_least_two():
    X = np.array([[0.0, 0.0], [1.0, 1.0]])

    with pytest.raises(ValueError, match="k >= 2"):
        EllipseCloud.from_local_cov(X, k=1)


def test_local_cov_requires_two_point_neighbourhood():
    X = np.array([[0.0, 0.0]])

    with pytest.raises(ValueError, match="at least two points"):
        EllipseCloud.from_local_cov(X, k=2)


def test_local_cov_merges_permuted_neighbourhoods():
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, np.sqrt(3.0) / 2.0],
        ]
    )

    ellcloud = EllipseCloud.from_local_cov(X, k=3)

    assert ellcloud.nbd.shape == (1, 3)
    assert np.array_equal(ellcloud.nbd[0], np.array([0, 1, 2]))


def test_local_cov_merges_identical_neighbourhoods():
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, np.sqrt(3.0) / 2.0],
            [5.0, 0.0],
            [6.0, 0.0],
            [5.5, np.sqrt(3.0) / 2.0],
        ]
    )

    ellcloud = EllipseCloud.from_local_cov(X, k=3)

    assert ellcloud.nbd.shape == (2, 3)
    assert np.array_equal(ellcloud.nbd, np.array([[0, 1, 2], [3, 4, 5]]))
    assert np.array_equal(ellcloud.nbd, np.sort(ellcloud.nbd, axis=1))


def test_ellipse_cloud_records_dimension_and_guards_plot():
    rng = np.random.default_rng(0)
    means = rng.standard_normal((4, 3))
    mats = rng.standard_normal((4, 3, 3))
    covs = np.empty((4, 3, 3))
    for idx in range(4):
        covs[idx] = mats[idx] @ mats[idx].T + np.eye(3)
    coefs = coef_from_cov(means, covs)
    cloud = EllipseCloud(coefs, means, covs, k=2, nbd=np.zeros((4, 0), dtype=int))
    assert cloud.n_dim == 3
    with pytest.raises(NotImplementedError):
        cloud.plot()
    with pytest.raises(NotImplementedError):
        cloud.rescale()
