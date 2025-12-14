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


def test_coef_from_cov_single_sample_shape():
    cov = np.array([[2.0, 0.5], [0.5, 1.0]])
    coef = coef_from_cov([0.0, 1.0], cov)
    assert coef.shape[0] == 1
    assert coef.ndim == 2


def test_conic_pack_unpack_roundtrip():
    rng = np.random.default_rng(42)
    A = rng.random((3, 3))
    A = A @ A.T + np.eye(3)
    b = rng.standard_normal(3)
    c = rng.normal()
    coef = pack_conic(A, b, c)
    A_rec, b_rec, c_rec = unpack_conic(coef)
    np.testing.assert_allclose(A, A_rec)
    np.testing.assert_allclose(b, b_rec)
    assert c == pytest.approx(c_rec)


@pytest.mark.parametrize("length, expected", [(6, 2), (10, 3), (15, 4)])
def test_infer_dim_from_coef_length(length, expected):
    assert infer_dim_from_coef_length(length) == expected


def test_coef_from_cov_general_dimension():
    rng = np.random.default_rng(123)
    center = rng.standard_normal(3)
    mat = rng.standard_normal((3, 3))
    cov = mat @ mat.T + np.eye(3)
    coef = coef_from_cov(center, cov)
    A, b, c = unpack_conic(coef)
    np.testing.assert_allclose(A[0], np.linalg.inv(cov))
    np.testing.assert_allclose(b[0], -A[0] @ center)
    assert c[0] == pytest.approx(center @ A[0] @ center)
