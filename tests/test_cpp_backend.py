from __future__ import annotations

import types

import numpy
import pytest

import ellphi._tangency_cpp as _cpp
from ellphi import build_info, cpp_linalg_kind
from ellphi.geometry import coef_from_cov
from ellphi.solver import (
    has_cpp_backend,
    pdist_tangency,
    tangency,
    tangency_python,
)


def _sample_coef(center, cov):
    return coef_from_cov(numpy.asarray(center), numpy.asarray(cov))


@pytest.fixture(scope="module")
def example_coefficients():
    p = _sample_coef([0.1, -0.2], [[0.6, 0.1], [0.1, 0.4]])
    q = _sample_coef([0.8, 0.3], [[0.9, -0.05], [-0.05, 0.5]])
    r = _sample_coef([-0.4, 0.5], [[0.7, 0.2], [0.2, 0.6]])
    return numpy.stack([p, q, r], axis=0)


def test_build_info():
    info = build_info()
    assert info.version
    assert info.backend_default == "auto"
    assert info.backend_choices == ("auto", "python", "cpp")
    assert info.cpp_backend_available == has_cpp_backend()
    assert info.cpp_linalg_kind == cpp_linalg_kind()
    if has_cpp_backend():
        assert info.cpp_linalg_kind in {"eigen", "internal"}
        assert info.cpp_backend_version is not None
    else:
        assert info.cpp_linalg_kind is None
        assert info.cpp_backend_version is None


def test_cpp_linalg_kind_missing_backend(monkeypatch):
    monkeypatch.setattr(_cpp, "_LIB", None)
    assert _cpp.is_available() is False
    assert _cpp.linalg_kind() is None
    assert _cpp.backend_version() is None


def test_cpp_linalg_kind_missing_symbol(monkeypatch):
    monkeypatch.setattr(_cpp, "_LIB", types.SimpleNamespace())
    with pytest.raises(RuntimeError, match="linear algebra metadata"):
        _cpp.linalg_kind()


def test_cpp_linalg_kind_empty_value(monkeypatch):
    def _linalg_kind():
        return None

    monkeypatch.setattr(
        _cpp, "_LIB", types.SimpleNamespace(tangency_linalg_kind=_linalg_kind)
    )
    with pytest.raises(RuntimeError, match="empty linear algebra kind"):
        _cpp.linalg_kind()


def test_cpp_linalg_kind_dummy_value(monkeypatch):
    def _linalg_kind():
        return b"eigen"

    monkeypatch.setattr(
        _cpp, "_LIB", types.SimpleNamespace(tangency_linalg_kind=_linalg_kind)
    )
    assert _cpp.linalg_kind() == "eigen"


@pytest.mark.skipif(not has_cpp_backend(), reason="C++ backend not available")
def test_tangency_cpp_matches_python(example_coefficients):
    p, q, _ = example_coefficients
    res_py = tangency_python(p, q)
    res_cpp = tangency(p, q, backend="cpp")
    numpy.testing.assert_allclose(res_cpp.t, res_py.t, rtol=0, atol=1e-12)
    numpy.testing.assert_allclose(res_cpp.point, res_py.point, rtol=0, atol=1e-12)
    numpy.testing.assert_allclose(res_cpp.mu, res_py.mu, rtol=0, atol=1e-12)


@pytest.mark.skipif(not has_cpp_backend(), reason="C++ backend not available")
def test_pdist_cpp_matches_python(example_coefficients):
    res_py = pdist_tangency(example_coefficients, parallel=False, backend="python")
    res_cpp = pdist_tangency(example_coefficients, parallel=False, backend="cpp")
    numpy.testing.assert_allclose(res_cpp, res_py, rtol=0, atol=1e-12)


@pytest.fixture(scope="module")
def example_coefficients_3d():
    p = _sample_coef(
        [0.1, -0.2, 0.3], [[0.6, 0.1, 0.05], [0.1, 0.4, 0.02], [0.05, 0.02, 0.5]]
    )
    q = _sample_coef(
        [0.8, 0.3, -0.1], [[0.9, -0.05, 0.1], [-0.05, 0.5, -0.03], [0.1, -0.03, 0.8]]
    )
    r = _sample_coef(
        [-0.4, 0.5, 0.0], [[0.7, 0.2, -0.1], [0.2, 0.6, 0.15], [-0.1, 0.15, 0.9]]
    )
    return numpy.stack([p, q, r], axis=0)


@pytest.mark.skipif(not has_cpp_backend(), reason="C++ backend not available")
def test_tangency_cpp_matches_python_3d(example_coefficients_3d):
    p, q, _ = example_coefficients_3d
    res_py = tangency_python(p, q)
    res_cpp = tangency(p, q, backend="cpp")
    numpy.testing.assert_allclose(res_cpp.t, res_py.t, rtol=0, atol=1e-12)
    numpy.testing.assert_allclose(res_cpp.point, res_py.point, rtol=0, atol=1e-12)
    numpy.testing.assert_allclose(res_cpp.mu, res_py.mu, rtol=0, atol=1e-12)


@pytest.mark.skipif(not has_cpp_backend(), reason="C++ backend not available")
def test_pdist_cpp_matches_python_3d(example_coefficients_3d):
    res_py = pdist_tangency(example_coefficients_3d, parallel=False, backend="python")
    res_cpp = pdist_tangency(example_coefficients_3d, parallel=False, backend="cpp")
    numpy.testing.assert_allclose(res_cpp, res_py, rtol=0, atol=1e-12)
