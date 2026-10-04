"""Factories shared across unit tests."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ellphi.ellcloud import EllipseCloud
from ellphi.geometry import coef_from_cov


def rotation_matrix(angle: float) -> NDArray[np.float64]:
    """Return a 2D rotation matrix for the given angle in radians."""
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[cos, -sin], [sin, cos]], dtype=float)


def random_covariance(rng: np.random.Generator, dim: int = 2) -> NDArray[np.float64]:
    """Draw a random symmetric positive-definite covariance matrix."""
    if dim == 2:
        axes = rng.uniform(1.0, 6.0, size=2)
        rot = rotation_matrix(rng.uniform(0.0, np.pi))
        return rot @ np.diag(axes) @ rot.T
    mat = rng.standard_normal((dim, dim))
    cov = mat @ mat.T
    return cov + dim * np.eye(dim, dtype=float)


def random_coef_pair(
    rng: np.random.Generator,
    *,
    dim: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Return two random ellipsoid coefficients using consistent sampling."""
    means = rng.uniform(-50.0, 50.0, size=(2, dim))
    covs = np.stack([random_covariance(rng, dim=dim) for _ in range(2)])
    coefs = coef_from_cov(means, covs)
    return coefs[0], coefs[1]


def random_cloud(
    rng: np.random.Generator, n_ellipses: int, *, dim: int = 2
) -> EllipseCloud:
    """Construct an ``EllipseCloud`` populated with random ellipsoids."""
    means = rng.uniform(-50.0, 50.0, size=(n_ellipses, dim))
    covs = np.stack([random_covariance(rng, dim=dim) for _ in range(n_ellipses)])
    coefs = coef_from_cov(means, covs)
    dummy_nbd = np.empty((n_ellipses, 0), dtype=int)
    return EllipseCloud(coef=coefs, mean=means, cov=covs, k=0, nbd=dummy_nbd)


def random_simplex(
    k: int,
    d: int,
    *,
    rng: np.random.Generator,
    cond_bound: float = 10.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Random centers and SPD matrices with ``cond(A_i) <= cond_bound``.

    Same sampling as ellcech's ``benchmarks.make_random_simplex``.
    """
    centers = rng.standard_normal((k, d))
    matrices = np.empty((k, d, d))
    for i in range(k):
        U = np.linalg.qr(rng.standard_normal((d, d)))[0]
        eigvals = rng.uniform(1.0, cond_bound, size=d)
        matrices[i] = U @ np.diag(eigvals) @ U.T
    return matrices, centers


def minimax_public_surrogate() -> tuple[np.ndarray, np.ndarray]:
    """Return the public-safe deterministic minimax regression instance.

    The literals are reproduced from this complete recipe. Create
    ``rng = np.random.default_rng(10602)`` and traverse zero-based instances
    ``0`` through ``12`` in order. For each instance, first traverse matrix
    rows ``i = 0`` through ``5`` and, for each row, draw
    ``q = np.linalg.qr(rng.standard_normal((2, 2)))[0]`` followed by
    ``eigenvalues = np.exp(rng.uniform(-1.0, 1.0, size=2))`` and construct
    ``(q * eigenvalues) @ q.T``. After all six matrices, draw
    ``centers = rng.standard_normal((6, 2))``. The returned literals are the
    matrices and centers from instance 12; this explicit matrix-first,
    row-major traversal is part of the fixture definition.
    """
    matrices = np.array(
        [
            [
                [0.6508813596346038, 0.2489660729149641],
                [0.2489660729149641, 0.7881614960406568],
            ],
            [
                [0.7004122505872777, 0.14197229432398561],
                [0.14197229432398561, 1.1445419752575752],
            ],
            [
                [0.6661363125679673, -0.7228217513541597],
                [-0.7228217513541597, 2.3994411001217943],
            ],
            [
                [0.555802471026214, 0.26197345221646695],
                [0.26197345221646695, 0.9029253520499695],
            ],
            [
                [2.304214987069779, 0.6449129856946328],
                [0.6449129856946328, 0.737837149118542],
            ],
            [
                [0.7373374529979612, 0.10773179234057872],
                [0.10773179234057872, 0.8413822561278285],
            ],
        ],
        dtype=float,
    )
    centers = np.array(
        [
            [0.3708705067926382, 0.48366396611109785],
            [1.16738707134914, -0.9051617225542165],
            [-0.6339208703399198, -0.47057481821754643],
            [0.1207125164253454, -1.267286300475832],
            [0.877114694674914, -0.46523959355638184],
            [-0.4268970087617493, 0.6371793650807637],
        ],
        dtype=float,
    )
    return matrices, centers
