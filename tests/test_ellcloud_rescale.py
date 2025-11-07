"""Regression tests for :mod:`ellphi.ellcloud`."""

from __future__ import annotations

import numpy as np
import pytest

from .factories import random_cloud


def test_rescale_returns_float_and_updates_arrays():
    """``EllipseCloud.rescale`` returns a float and rescales members."""

    rng = np.random.default_rng(2024)
    cloud = random_cloud(rng, n_ellipses=5)

    cov_before = cloud.cov.copy()
    coef_before = cloud.coef.copy()

    scale = cloud.rescale()

    assert isinstance(scale, float)
    np.testing.assert_allclose(cloud.cov, cov_before / scale**2)
    np.testing.assert_allclose(cloud.coef, coef_before * scale**2)


def test_rescale_not_implemented_for_three_dimensions():
    rng = np.random.default_rng(2025)
    cloud = random_cloud(rng, n_ellipses=3, dim=3)

    with pytest.raises(NotImplementedError):
        cloud.rescale()
