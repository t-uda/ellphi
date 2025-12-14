"""Tests for the ellphi visualization module."""

import numpy as np
from matplotlib.patches import Ellipse
from ellphi.visualization import ellipse_patch


def test_ellipse_patch_from_cov():
    """Test creating an ellipse patch from a covariance matrix."""
    center = np.array([1.0, 2.0])
    cov = np.array([[0.2, 0.1], [0.1, 0.3]])
    patch = ellipse_patch(center, cov=cov)

    assert isinstance(patch, Ellipse)
    assert np.allclose(patch.center, center)

    # Check that width, height, and angle are set correctly
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    r_major = np.sqrt(eigenvalues[1])
    r_minor = np.sqrt(eigenvalues[0])
    theta = np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1])

    assert np.isclose(patch.width, 2 * r_major)
    assert np.isclose(patch.height, 2 * r_minor)
    assert np.isclose(patch.angle, np.degrees(theta))


def test_ellipse_patch_direct_params():
    """Test creating an ellipse patch with direct parameters."""
    center = np.array([0.0, 0.0])
    r_major = 2.0
    r_minor = 1.0
    theta = np.pi / 4
    patch = ellipse_patch(center, r_major=r_major, r_minor=r_minor, theta=theta)

    assert isinstance(patch, Ellipse)
    assert np.allclose(patch.center, center)
    assert np.isclose(patch.width, 2 * r_major)
    assert np.isclose(patch.height, 2 * r_minor)
    assert np.isclose(patch.angle, np.degrees(theta))
