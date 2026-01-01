import numpy as np
import pytest

from ellphi.geometry import (
    coef_from_cov,
    infer_dim_from_coef_length,
    pack_conic,
    unpack_conic,
    unpack_single_conic,
)


def test_infer_dim_from_coef_length_rejects_invalid_discriminant():
    with pytest.raises(ValueError, match="symmetric quadratic form"):
        infer_dim_from_coef_length(7)


def test_pack_conic_requires_square_matrix():
    matrix = np.zeros((2, 3))
    linear = np.zeros(3)
    with pytest.raises(ValueError, match="Quadratic matrices must have shape"):
        pack_conic(matrix, linear, 0.0)


def test_pack_conic_requires_linear_match():
    matrix = np.eye(2)
    linear = np.zeros(3)
    with pytest.raises(ValueError, match="Linear term incompatible"):
        pack_conic(matrix, linear, 0.0)


def test_unpack_conic_rejects_invalid_rank():
    coef = np.zeros((2, 2, 2))
    with pytest.raises(ValueError, match="one- or two-dimensional"):
        unpack_conic(coef)


def test_unpack_single_conic_rejects_multiple_entries():
    centers = np.array([[0.0, 0.0], [1.0, 1.0]])
    covs = np.stack([np.eye(2), 2.0 * np.eye(2)])
    coef = coef_from_cov(centers, covs)
    with pytest.raises(ValueError, match="Expected coefficients for a single conic"):
        unpack_single_conic(coef)


@pytest.mark.parametrize(
    "centers, covs, message",
    [
        (np.zeros((2, 2)), np.zeros((1, 2, 2)), "number of centres"),
        (np.zeros((1, 2)), np.zeros((1, 2, 3)), "Covariance matrices must be square"),
        (np.zeros((1, 3)), np.zeros((1, 2, 2)), "dimensionality"),
    ],
)
def test_coef_from_cov_validates_shapes(centers, covs, message):
    with pytest.raises(ValueError, match=message):
        coef_from_cov(centers, covs)


def test_coef_from_cov_singular_returns_nan():
    center = np.array([0.0, 0.0])
    cov = np.array([[1.0, 0.0], [0.0, 0.0]])
    coef = coef_from_cov(center, cov)
    assert np.isnan(coef).all()
