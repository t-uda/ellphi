import types
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import numpy as np
import pytest

import ellphi._tangency_cpp as _cpp


def test_library_suffix_prefers_sysconfig(monkeypatch):
    monkeypatch.setattr(_cpp.sysconfig, "get_config_var", lambda _: ".pyd")
    assert _cpp._library_suffix() == ".pyd"


@pytest.mark.parametrize(
    "platform, expected",
    [
        ("win32", ".dll"),
        ("darwin", ".dylib"),
        ("linux", ".so"),
    ],
)
def test_library_suffix_platform_fallback(monkeypatch, platform, expected):
    monkeypatch.setattr(_cpp.sysconfig, "get_config_var", lambda _: None)
    monkeypatch.setattr(_cpp.sys, "platform", platform)
    assert _cpp._library_suffix() == expected


def test_expected_backend_version_uses_package_metadata(monkeypatch):
    monkeypatch.setattr(_cpp, "__version__", "0+unknown")
    monkeypatch.setattr(_cpp, "package_version", lambda _: "9.9.9")
    assert _cpp._expected_backend_version() == "9.9.9"


def test_expected_backend_version_falls_back_on_missing_package(monkeypatch):
    monkeypatch.setattr(_cpp, "__version__", "0+unknown")

    def raise_not_found(_: str) -> str:
        raise PackageNotFoundError

    monkeypatch.setattr(_cpp, "package_version", raise_not_found)
    assert _cpp._expected_backend_version() == "0+unknown"


def test_library_version_requires_metadata():
    with pytest.raises(RuntimeError, match="version metadata"):
        _cpp._library_version(types.SimpleNamespace())


def test_library_version_rejects_empty_value():
    def version_func():
        return None

    lib = types.SimpleNamespace(tangency_backend_version=version_func)
    with pytest.raises(RuntimeError, match="empty version string"):
        _cpp._library_version(lib)


def test_library_version_decodes_bytes():
    def version_func():
        return b"1.2.3"

    lib = types.SimpleNamespace(tangency_backend_version=version_func)
    assert _cpp._library_version(lib) == "1.2.3"


def test_validate_library_version_reports_mismatch(monkeypatch):
    monkeypatch.setattr(_cpp, "_expected_backend_version", lambda: "expected")
    monkeypatch.setattr(_cpp, "_library_version", lambda _: "actual")
    with pytest.raises(RuntimeError, match="version mismatch"):
        _cpp._validate_library_version(object())


def test_load_library_raises_on_missing_file(monkeypatch, tmp_path):
    missing = tmp_path / "missing.so"
    monkeypatch.setattr(_cpp, "_library_path", lambda: Path(missing))
    with pytest.raises(FileNotFoundError, match="library missing"):
        _cpp._load_library()


def test_ensure_available_uses_error_message(monkeypatch):
    monkeypatch.setattr(_cpp, "_LIB", None)
    monkeypatch.setattr(_cpp, "_LIB_ERROR", "boom")
    with pytest.raises(RuntimeError, match="boom"):
        _cpp._ensure_available()


def test_ensure_available_defaults_message(monkeypatch):
    monkeypatch.setattr(_cpp, "_LIB", None)
    monkeypatch.setattr(_cpp, "_LIB_ERROR", None)
    with pytest.raises(RuntimeError, match="C\\+\\+ backend not available"):
        _cpp._ensure_available()


@pytest.mark.parametrize(
    "message, exc_type",
    [
        ("x0 must be provided for Newton method", ValueError),
        ("Degenerate conic", ZeroDivisionError),
        ("other", RuntimeError),
        ("", RuntimeError),
    ],
)
def test_raise_backend_error(message, exc_type):
    with pytest.raises(exc_type):
        _cpp._raise_backend_error(message)


def test_tangency_rejects_non_1d_inputs(monkeypatch):
    def tangency_solve(*_args, **_kwargs):
        return 0

    dummy_lib = types.SimpleNamespace(tangency_solve=tangency_solve)
    monkeypatch.setattr(_cpp, "_ensure_available", lambda: dummy_lib)
    pcoef = np.zeros((1, 6))
    qcoef = np.zeros((6,))
    with pytest.raises(ValueError, match="one-dimensional"):
        _cpp.tangency(
            pcoef,
            qcoef,
            method="brentq",
            bracket=(0.0, 1.0),
            x0=None,
            hybrid_bracket_maxiter=1,
            hybrid_newton_maxiter=1,
            failsafe=True,
        )


def test_tangency_rejects_shape_mismatch(monkeypatch):
    def tangency_solve(*_args, **_kwargs):
        return 0

    dummy_lib = types.SimpleNamespace(tangency_solve=tangency_solve)
    monkeypatch.setattr(_cpp, "_ensure_available", lambda: dummy_lib)
    pcoef = np.zeros((6,))
    qcoef = np.zeros((7,))
    with pytest.raises(ValueError, match="same length"):
        _cpp.tangency(
            pcoef,
            qcoef,
            method="brentq",
            bracket=(0.0, 1.0),
            x0=None,
            hybrid_bracket_maxiter=1,
            hybrid_newton_maxiter=1,
            failsafe=True,
        )


def test_pdist_tangency_rejects_invalid_shape(monkeypatch):
    def pdist_tangency(*_args, **_kwargs):
        return 0

    dummy_lib = types.SimpleNamespace(pdist_tangency=pdist_tangency)
    monkeypatch.setattr(_cpp, "_ensure_available", lambda: dummy_lib)
    coef = np.zeros((6,))
    with pytest.raises(ValueError, match="shape \\(m, n\\)"):
        _cpp.pdist_tangency(coef)
