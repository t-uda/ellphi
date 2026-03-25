import numpy as np
import pytest
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform

from ellphi.ellcloud import EllipseCloud
from ellphi.geometry import coef_from_axes, coef_from_cov
from ellphi.solver import pdist_tangency

from .factories import random_cloud


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


def test_post_init_value_errors():
    coef = np.zeros((2, 5))
    mean = np.zeros((2, 2))
    cov = np.zeros((2, 2, 2))
    nbd = np.zeros((2, 2))
    with pytest.raises(ValueError):
        EllipseCloud(coef.ravel(), mean, cov, 2, nbd)
    with pytest.raises(ValueError):
        EllipseCloud(coef, np.zeros((3, 2)), cov, 2, nbd)
    with pytest.raises(ValueError):
        EllipseCloud(coef, mean, np.zeros((2, 3, 3)), 2, nbd)
    with pytest.raises(ValueError):
        EllipseCloud(coef, mean, cov, 2, np.zeros((3, 2)))


def test_post_init_validates_mean_shape():
    coef = np.stack(
        [
            coef_from_axes([0.0, 0.0], 1.0, 1.0, 0.0),
            coef_from_axes([1.0, 0.5], 1.5, 1.2, 0.3),
        ],
        axis=0,
    )
    mean = np.zeros((3, 2))
    cov = np.stack([np.eye(2), np.eye(2)], axis=0)
    nbd = np.zeros((2, 0), dtype=int)
    with pytest.raises(ValueError, match="Mean array has shape"):
        EllipseCloud(coef, mean, cov, k=0, nbd=nbd)


def test_post_init_validates_cov_shape():
    coef = np.stack(
        [
            coef_from_axes([0.0, 0.0], 1.0, 1.0, 0.0),
            coef_from_axes([1.0, 0.5], 1.5, 1.2, 0.3),
        ],
        axis=0,
    )
    mean = np.zeros((2, 2))
    cov = np.zeros((2, 3, 3))
    nbd = np.zeros((2, 0), dtype=int)
    with pytest.raises(ValueError, match="Covariance array has shape"):
        EllipseCloud(coef, mean, cov, k=0, nbd=nbd)


def test_post_init_validates_neighbourhood_shape():
    coef = np.stack(
        [
            coef_from_axes([0.0, 0.0], 1.0, 1.0, 0.0),
            coef_from_axes([1.0, 0.5], 1.5, 1.2, 0.3),
        ],
        axis=0,
    )
    mean = np.zeros((2, 2))
    cov = np.stack([np.eye(2), np.eye(2)], axis=0)
    nbd = np.zeros((3, 1), dtype=int)
    with pytest.raises(ValueError, match="Neighbourhood index array"):
        EllipseCloud(coef, mean, cov, k=0, nbd=nbd)


def test_iter_returns_coefficients():
    coef = np.stack(
        [
            coef_from_axes([0.0, 0.0], 1.0, 1.0, 0.0),
            coef_from_axes([1.0, 0.5], 1.5, 1.2, 0.3),
        ],
        axis=0,
    )
    mean = np.zeros((2, 2))
    cov = np.stack([np.eye(2), np.eye(2)], axis=0)
    nbd = np.zeros((2, 0), dtype=int)
    cloud = EllipseCloud(coef, mean, cov, k=0, nbd=nbd)
    first = next(iter(cloud))
    np.testing.assert_allclose(first, coef[0])


def test_plot_creates_axes_and_patches():
    coef = np.stack(
        [
            coef_from_axes([0.0, 0.0], 1.0, 1.0, 0.0),
            coef_from_axes([1.0, 0.5], 1.5, 1.2, 0.3),
        ],
        axis=0,
    )
    mean = np.array([[0.0, 0.0], [1.0, 0.5]])
    cov = np.stack([np.eye(2), np.eye(2)], axis=0)
    nbd = np.zeros((2, 0), dtype=int)
    cloud = EllipseCloud(coef, mean, cov, k=0, nbd=nbd)
    ax = cloud.plot()
    assert len(ax.patches) == 2
    plt.close(ax.figure)


def test_from_cov_accepts_single_2d_sample():
    center = np.array([1.0, 2.0])
    cov = np.array([[2.0, 0.1], [0.1, 1.5]])

    cloud = EllipseCloud.from_cov(center, cov)

    assert cloud.k == 0
    assert cloud.n == 1
    assert cloud.nbd.shape == (1, 0)
    np.testing.assert_allclose(cloud.mean, center[np.newaxis, :])
    np.testing.assert_allclose(cloud.cov, cov[np.newaxis, :, :])
    np.testing.assert_allclose(cloud.coef, coef_from_cov(center, cov))


def test_from_cov_matches_manual_constructor_3d(rng):
    expected = random_cloud(rng, n_ellipses=4, dim=3)

    cloud = EllipseCloud.from_cov(expected.mean, expected.cov)

    assert cloud.k == 0
    assert cloud.nbd.shape == (expected.n, 0)
    np.testing.assert_allclose(cloud.mean, expected.mean)
    np.testing.assert_allclose(cloud.cov, expected.cov)
    np.testing.assert_allclose(cloud.coef, expected.coef)


@pytest.mark.parametrize("method", ["median", "average"])
def test_from_cov_rescaling_matches_manual_rescale(rng, method):
    original = random_cloud(rng, n_ellipses=5)
    manual = EllipseCloud(
        coef=original.coef.copy(),
        mean=original.mean.copy(),
        cov=original.cov.copy(),
        k=0,
        nbd=np.empty((original.n, 0), dtype=int),
    )
    manual.rescale(method=method)

    cloud = EllipseCloud.from_cov(
        original.mean.copy(),
        original.cov.copy(),
        rescaling=method,
    )

    np.testing.assert_allclose(cloud.mean, manual.mean)
    np.testing.assert_allclose(cloud.cov, manual.cov)
    np.testing.assert_allclose(cloud.coef, manual.coef)


def test_pdist_tangency_wrapper_matches_solver(rng):
    cloud = random_cloud(rng, n_ellipses=4)
    wrapper = cloud.pdist_tangency(backend="python", parallel=False)
    direct = pdist_tangency(cloud, backend="python", parallel=False)
    np.testing.assert_allclose(wrapper, direct)


def test_distance_matrix_matches_squareform(rng):
    cloud = random_cloud(rng, n_ellipses=4)

    matrix = cloud.distance_matrix(backend="python", parallel=False)
    expected = squareform(cloud.pdist_tangency(backend="python", parallel=False))

    assert matrix.shape == (cloud.n, cloud.n)
    np.testing.assert_allclose(matrix, expected)


def test_str_method():
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    ellcloud = EllipseCloud.from_local_cov(X, k=3)
    s = str(ellcloud)
    assert "EllipseCloud" in s
    assert "coef=array<(1, 6)>" in s
    assert "mean=array<(1, 2)>" in s
    assert "cov=array<(1, 2, 2)>" in s
    assert "k=3" in s
    assert "nbd=array<(1, 3)>" in s


def test_from_point_cloud_rescaling():
    X = np.random.rand(10, 2)
    ellcloud_median = EllipseCloud.from_point_cloud(X, k=5, rescaling="median")
    ellcloud_avg = EllipseCloud.from_point_cloud(X, k=5, rescaling="average")

    assert ellcloud_median.cov.shape == (ellcloud_median.n, 2, 2)
    assert ellcloud_avg.cov.shape == (ellcloud_avg.n, 2, 2)
    assert ellcloud_median.n <= 10
    assert ellcloud_avg.n <= 10

    with pytest.raises(NotImplementedError):
        EllipseCloud.from_point_cloud(X, k=5, method="unknown")

    with pytest.raises(NotImplementedError):
        ellcloud_median.rescale(method="unknown")
