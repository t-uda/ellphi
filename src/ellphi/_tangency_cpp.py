"""Lazy loader for the optional C++ tangency backend."""

from __future__ import annotations

import ctypes
import sysconfig
import threading
from pathlib import Path
from types import ModuleType
from typing import Optional

import numpy

__all__ = ["is_available", "load", "get_module", "get_error"]

_LOCK = threading.Lock()
_MODULE: ModuleType | None = None
_ERROR: Exception | None = None


class _TangencyResult(ctypes.Structure):
    _fields_ = [
        ("t", ctypes.c_double),
        ("point_x", ctypes.c_double),
        ("point_y", ctypes.c_double),
        ("mu", ctypes.c_double),
    ]


def _shared_library_path() -> Path:
    source = Path(__file__).with_name("_tangency_cpp_impl.cpp")
    suffix = sysconfig.get_config_var("SHLIB_SUFFIX") or ".so"
    return source.with_suffix(suffix)


def _error_from_code(code: int) -> Exception:
    if code == 1:
        return ZeroDivisionError("Degenerate conic (determinant zero)")
    if code == 2:
        return ValueError("Root is not bracketed for the selected interval")
    if code == 3:
        return ValueError("Unknown method")
    if code == 4:
        return ValueError("x0 must be provided for Newton method")
    if code == 5:
        return ZeroDivisionError("Zero derivative encountered in Newton method")
    if code == 6:
        return ValueError("Bracket must satisfy a < b")
    return RuntimeError(f"Tangency solver failed with error code {code}")


def _wrap_library(lib: ctypes.CDLL) -> ModuleType:
    tangency_solver = lib.tangency_solver
    tangency_solver.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_double,
        ctypes.c_int,
        ctypes.POINTER(_TangencyResult),
    ]
    tangency_solver.restype = ctypes.c_int

    pdist_solver = lib.pdist_tangency_solver
    pdist_solver.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int64,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    pdist_solver.restype = ctypes.c_int

    def tangency(
        pcoef: numpy.ndarray,
        qcoef: numpy.ndarray,
        *,
        method: str = "brentq+newton",
        bracket: tuple[float, float] = (0.0, 1.0),
        x0: float | None = None,
    ) -> tuple[float, numpy.ndarray, float]:
        p_arr = numpy.ascontiguousarray(pcoef, dtype=float)
        q_arr = numpy.ascontiguousarray(qcoef, dtype=float)
        if p_arr.shape != (6,) or q_arr.shape != (6,):
            raise ValueError("Coefficient arrays must have shape (6,)")

        bracket_arr = numpy.ascontiguousarray(bracket, dtype=float)
        if bracket_arr.shape != (2,):
            raise ValueError("Bracket must be a pair of floats")

        result = _TangencyResult()
        status = tangency_solver(
            p_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            q_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            method.encode("ascii"),
            bracket_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            0.0 if x0 is None else float(x0),
            0 if x0 is None else 1,
            ctypes.byref(result),
        )
        if status != 0:
            raise _error_from_code(status)

        point = numpy.array([result.point_x, result.point_y], dtype=float)
        return float(result.t), point, float(result.mu)

    def pdist_tangency(
        coefficients: numpy.ndarray,
        *,
        method: str = "brentq+newton",
        bracket: tuple[float, float] = (0.0, 1.0),
    ) -> numpy.ndarray:
        coef_arr = numpy.ascontiguousarray(coefficients, dtype=float)
        if coef_arr.ndim != 2 or coef_arr.shape[1] != 6:
            raise ValueError("Coefficient matrix must have shape (N, 6)")

        m = int(coef_arr.shape[0])
        out = numpy.empty(m * (m - 1) // 2, dtype=float)
        bracket_arr = numpy.ascontiguousarray(bracket, dtype=float)
        if bracket_arr.shape != (2,):
            raise ValueError("Bracket must be a pair of floats")

        status = pdist_solver(
            coef_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int64(m),
            method.encode("ascii"),
            bracket_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if status != 0:
            raise _error_from_code(status)
        return out

    module = ModuleType("_tangency_cpp_impl")
    module.tangency = tangency  # type: ignore[attr-defined]
    module.pdist_tangency = pdist_tangency  # type: ignore[attr-defined]
    return module


def load() -> ModuleType:
    """Import and return the compiled C++ module, raising on failure."""
    global _MODULE, _ERROR
    if _MODULE is not None:
        return _MODULE
    if _ERROR is not None:
        raise _ERROR

    with _LOCK:
        if _MODULE is not None:
            return _MODULE
        if _ERROR is not None:
            raise _ERROR
        try:
            library_path = _shared_library_path()
            lib = ctypes.CDLL(str(library_path))
            _MODULE = _wrap_library(lib)
        except OSError as exc:  # pragma: no cover - exercised via Python fallback
            library_str = str(library_path)
            error = ImportError(
                "Compiled C++ tangency backend is missing. "
                f"Expected shared library at '{library_str}'."
            )
            error.__cause__ = exc
            _ERROR = error
            raise error
    return _MODULE


def get_module() -> Optional[ModuleType]:
    """Return the compiled module if available; otherwise ``None``."""
    try:
        return load()
    except Exception:  # pragma: no cover - exercised via Python fallback
        return None


def is_available() -> bool:
    """Return ``True`` if the compiled backend can be imported."""
    return get_module() is not None


def get_error() -> Exception | None:
    """Return the import error encountered when loading the C++ backend."""
    return _ERROR
