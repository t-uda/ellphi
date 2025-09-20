import numpy as np
import pytest

from ellphi.ellcloud import EllipseCloud


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
