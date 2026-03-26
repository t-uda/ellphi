"""Regression tests for `ellphi.ellcloud`."""

from __future__ import annotations

import numpy as np

from ellphi import RescaleDiagnostics

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


def test_rescale_diagnostics_type_and_scale_consistency():
    """``rescale(return_diagnostics=True)`` returns a RescaleDiagnostics."""

    rng = np.random.default_rng(2025)
    cloud = random_cloud(rng, n_ellipses=8)

    diag = cloud.rescale(return_diagnostics=True)

    assert isinstance(diag, RescaleDiagnostics)
    assert isinstance(diag.scale, float)
    assert diag.pre_summary.shape == (2,)
    assert diag.post_summary.shape == (2,)


def test_rescale_diagnostics_pre_post_relationship():
    """post_summary equals pre_summary / scale."""

    rng = np.random.default_rng(2026)
    cloud = random_cloud(rng, n_ellipses=10)

    diag = cloud.rescale(return_diagnostics=True)

    np.testing.assert_allclose(diag.post_summary, diag.pre_summary / diag.scale)


def test_rescale_default_matches_diagnostics_scale():
    """Default (no diagnostics) scale matches diagnostics.scale."""

    rng = np.random.default_rng(2027)
    cloud_a = random_cloud(rng, n_ellipses=6)

    rng2 = np.random.default_rng(2027)
    cloud_b = random_cloud(rng2, n_ellipses=6)

    scale = cloud_a.rescale()
    diag = cloud_b.rescale(return_diagnostics=True)

    assert scale == diag.scale
