import numpy as np
from numpy.typing import NDArray

from ellphi.ellcloud import EllipseCloud
from ellphi.geometry import coef_from_cov


def rotation_matrix(angle: float) -> NDArray[np.float64]:
    """Return a 2D rotation matrix for the given angle in radians."""
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[cos, -sin], [sin, cos]], dtype=float)


def random_covariance(rng: np.random.Generator, dim: int = 2) -> NDArray[np.float64]:
    """Draw a random, symmetric positive-definite covariance matrix."""
    if dim == 2:
        axes = rng.uniform(1.0, 6.0, size=2)
        rot = rotation_matrix(rng.uniform(0.0, np.pi))
        return rot @ np.diag(axes) @ rot.T
    mat = rng.normal(size=(dim, dim))
    cov = mat @ mat.T
    cov += dim * np.eye(dim)
    return cov


def random_coef_pair(
    rng: np.random.Generator, *, dim: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Return two random ellipsoid coefficients using consistent sampling."""
    means = rng.uniform(-5.0, 5.0, size=(2, dim))
    covs = np.stack([random_covariance(rng, dim=dim) for _ in range(2)])
    coefs = coef_from_cov(means, covs)
    return coefs[0], coefs[1]


def random_cloud(
    rng: np.random.Generator, *, n_ellipses: int, dim: int = 2
) -> EllipseCloud:
    """Construct an ``EllipseCloud`` populated with random ellipsoids."""
    means = rng.uniform(-10.0, 10.0, size=(n_ellipses, dim))
    covs = np.stack([random_covariance(rng, dim=dim) for _ in range(n_ellipses)])
    coefs = coef_from_cov(means, covs)
    dummy_nbd = np.empty((n_ellipses, 0), dtype=int)
    return EllipseCloud(coef=coefs, mean=means, cov=covs, k=0, nbd=dummy_nbd)
