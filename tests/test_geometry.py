
import numpy as np
import pytest

from ellphi.geometry import (
    unit_vector,
    axes_from_cov,
    coef_from_axes,
    coef_from_array,
    coef_from_cov,
)

# ------------------------------------------------------------
# 1. unit_vector basics
# ------------------------------------------------------------
@pytest.mark.parametrize(
    "theta, expected",
    [
        (0.0, (1.0, 0.0)),
        (np.pi / 2, (0.0, 1.0)),
        (np.pi, (-1.0, 0.0)),
        (3 * np.pi / 2, (0.0, -1.0)),
    ],
)
def test_unit_vector(theta, expected):
    v = unit_vector(theta)
    assert v.shape == (2,)
    assert np.allclose(v, expected, atol=1e-12)


# ------------------------------------------------------------
# 2. axes_from_cov gives r1 >= r2
# ------------------------------------------------------------
def test_axes_order():
    cov = np.array([[9.0, 0.0], [0.0, 1.0]])
    r1, r2, _ = axes_from_cov(cov)
    assert r1 >= r2, "r1 should be the major semi-axis"


# ------------------------------------------------------------
# 3. coef_from_axes vs. coef_from_array consistency
# ------------------------------------------------------------
def test_coef_scalar_vs_batch():
    rng = np.random.default_rng(42)
    params = rng.random((10, 5)) * np.array([2, 2, 1.5, 1.5, np.pi])
    coef_batch = coef_from_array(params)
    # Compare each row with scalar function
    for row, coef_row in zip(params, coef_batch):
        coef_single = coef_from_axes(*row)
        assert np.allclose(coef_single, coef_row, rtol=1e-12, atol=1e-12)


# ------------------------------------------------------------
# 4. coef_from_cov agrees with coef_from_axes
# ------------------------------------------------------------
def test_coef_from_cov():
    cov = np.array([[4.0, 1.2], [1.2, 3.0]])
    x0, y0 = 0.3, -0.8
    r1, r2, th = axes_from_cov(cov)
    coef1 = coef_from_axes(x0, y0, r1, r2, th)
    coef2 = coef_from_cov(x0, y0, cov)
    assert np.allclose(coef1, coef2, rtol=1e-12, atol=1e-12)

