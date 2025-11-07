import numpy as np
import pytest

from ellphi.geometry import (
    unit_vector,
    axes_from_cov,
    coef_from_axes,
    coef_from_cov,
    infer_dim_from_coef_length,
    pack_conic,
    unpack_conic,
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
# 3. coef_from_cov agrees with coef_from_axes
# ------------------------------------------------------------
def test_coef_from_cov():
    cov = np.array([[4.0, 1.2], [1.2, 3.0]])
    x0, y0 = 0.3, -0.8
    r1, r2, th = axes_from_cov(cov)
    coef1 = coef_from_axes([x0, y0], r1, r2, th)
    coef2 = coef_from_cov([x0, y0], cov)
    assert np.allclose(coef1, coef2, rtol=1e-12, atol=1e-12)


def test_pack_unpack_roundtrip_three_dimensional():
    quad = np.array(
        [[4.0, 1.0, 0.2], [1.0, 3.5, -0.4], [0.2, -0.4, 2.2]],
        dtype=float,
    )
    linear = np.array([-1.0, 0.5, 2.0], dtype=float)
    constant = 0.7
    coef = pack_conic(quad, linear, constant)
    quad_rec, linear_rec, constant_rec = unpack_conic(coef)
    np.testing.assert_allclose(quad_rec, quad)
    np.testing.assert_allclose(linear_rec, linear)
    assert constant_rec == pytest.approx(constant)


@pytest.mark.parametrize("length, expected", [(6, 2), (10, 3)])
def test_infer_dim_from_coef_length(length, expected):
    assert infer_dim_from_coef_length(length) == expected


def test_coef_from_cov_three_dimensional():
    rng = np.random.default_rng(0)
    center = rng.normal(size=3)
    basis = rng.normal(size=(3, 3))
    rot, _ = np.linalg.qr(basis)
    eigenvalues = rng.uniform(0.5, 2.0, size=3)
    cov = rot @ np.diag(eigenvalues**2) @ rot.T

    coef = coef_from_cov(center, cov)[0]
    quad, linear, constant = unpack_conic(coef)

    inv_cov = np.linalg.inv(cov)
    np.testing.assert_allclose(quad, inv_cov)
    np.testing.assert_allclose(linear, -inv_cov @ center)
    assert constant == pytest.approx(center @ inv_cov @ center)
