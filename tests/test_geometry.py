import numpy as np
import pytest

from ellphi.geometry import (
    axes_from_cov,
    coef_from_axes,
    coef_from_cov,
    infer_dim_from_coef_length,
    pack_conic,
    unpack_conic,
    unit_vector,
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
# 3. coef_from_cov agrees with matrix identities in any dimension
# ------------------------------------------------------------
@pytest.mark.parametrize("dim", [2, 3])
def test_coef_from_cov(dim):
    rng = np.random.default_rng(42)
    mean = rng.uniform(-2.0, 2.0, size=dim)
    mat = rng.normal(size=(dim, dim))
    cov = mat @ mat.T + np.eye(dim)

    coef = coef_from_cov(mean, cov)[0]
    A, b, c = unpack_conic(coef)

    expected_A = np.linalg.inv(cov)
    expected_b = -expected_A @ mean
    expected_c = mean @ expected_A @ mean

    np.testing.assert_allclose(A, expected_A, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(b, expected_b, rtol=1e-12, atol=1e-12)
    assert c == pytest.approx(expected_c, rel=1e-12, abs=1e-12)


# ------------------------------------------------------------
# 4. coef_from_cov matches coef_from_axes in 2D
# ------------------------------------------------------------
def test_coef_from_cov_matches_axes():
    cov = np.array([[4.0, 1.2], [1.2, 3.0]])
    x0, y0 = 0.3, -0.8
    r1, r2, th = axes_from_cov(cov)
    coef1 = coef_from_axes([x0, y0], r1, r2, th)
    coef2 = coef_from_cov([x0, y0], cov)[0]
    assert np.allclose(coef1, coef2, rtol=1e-12, atol=1e-12)


# ------------------------------------------------------------
# 5. Pack/unpack round-trip
# ------------------------------------------------------------
@pytest.mark.parametrize("dim", [2, 3, 5])
def test_pack_unpack_roundtrip(dim):
    rng = np.random.default_rng(123)
    A = rng.normal(size=(dim, dim))
    A = A @ A.T + np.eye(dim)
    b = rng.normal(size=dim)
    c = rng.normal()

    coef = pack_conic(A, b, c)
    A_rt, b_rt, c_rt = unpack_conic(coef)

    np.testing.assert_allclose(A_rt, A, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(b_rt, b, rtol=1e-12, atol=1e-12)
    assert c_rt == pytest.approx(c, rel=1e-12, abs=1e-12)


# ------------------------------------------------------------
# 6. infer_dim_from_coef_length recovers dimensionality
# ------------------------------------------------------------
@pytest.mark.parametrize("dim", [2, 3, 7])
def test_infer_dim_from_coef_length(dim):
    length = (dim + 1) * (dim + 2) // 2
    assert infer_dim_from_coef_length(length) == dim


def test_infer_dim_from_coef_length_rejects_invalid():
    with pytest.raises(ValueError):
        infer_dim_from_coef_length(5)
