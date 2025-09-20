from __future__ import annotations

"""Bindings for the pre-built C++ tangency backend."""

import ctypes
import sys
import sysconfig
from pathlib import Path
from typing import Tuple

import numpy

from ._solver_python import TangencyResult

_LIB_NAME = "_tangency_cpp_impl"
_ERROR_BUFFER = 4096


def _library_suffix() -> str:
    suffix = sysconfig.get_config_var("SHLIB_SUFFIX")
    if suffix:
        return suffix
    if sys.platform.startswith("win"):
        return ".dll"
    if sys.platform == "darwin":
        return ".dylib"
    return ".so"


def _library_path() -> Path:
    return Path(__file__).with_name(f"{_LIB_NAME}{_library_suffix()}")


def _load_library() -> ctypes.CDLL:
    lib_path = _library_path()
    if not lib_path.exists():
        raise FileNotFoundError(f"C++ backend library missing: {lib_path}")
    return ctypes.CDLL(str(lib_path))


_LIB: ctypes.CDLL | None
try:  # pragma: no cover - build process is environment dependent
    _LIB = _load_library()
except (OSError, FileNotFoundError):  # pragma: no cover
    _LIB = None


def is_available() -> bool:
    return _LIB is not None


def _ensure_available() -> ctypes.CDLL:
    if _LIB is None:
        raise RuntimeError(
            "C++ backend not available. Run 'python build_tangency_cpp.py' to build it."
        )
    return _LIB


def _raise_backend_error(message: str) -> None:
    if "x0 must be provided for Newton method" in message:
        raise ValueError(message)
    if "Degenerate conic" in message:
        raise ZeroDivisionError(message)
    raise RuntimeError(message or "Unknown C++ backend error")


def tangency(
    pcoef: numpy.ndarray,
    qcoef: numpy.ndarray,
    *,
    method: str,
    bracket: Tuple[float, float],
    x0: float | None,
) -> TangencyResult:
    lib = _ensure_available()

    func = lib.tangency_solve
    func.restype = ctypes.c_int
    func.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]

    pcoef_arr = numpy.ascontiguousarray(pcoef, dtype=float)
    qcoef_arr = numpy.ascontiguousarray(qcoef, dtype=float)
    bracket_arr = numpy.ascontiguousarray(bracket, dtype=float)

    t_out = ctypes.c_double()
    point_out = (ctypes.c_double * 2)()
    mu_out = ctypes.c_double()
    error_buffer = ctypes.create_string_buffer(_ERROR_BUFFER)

    has_x0 = 0 if x0 is None else 1
    x0_val = 0.0 if x0 is None else float(x0)

    status = func(
        pcoef_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        qcoef_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        method.encode("utf-8"),
        bracket_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(has_x0),
        ctypes.c_double(x0_val),
        ctypes.byref(t_out),
        point_out,
        ctypes.byref(mu_out),
        ctypes.cast(error_buffer, ctypes.c_void_p),
        ctypes.c_size_t(_ERROR_BUFFER),
    )

    if status != 0:  # pragma: no cover - propagated to Python layer
        message = error_buffer.value.decode("utf-8", errors="ignore")
        _raise_backend_error(message)

    point = numpy.ctypeslib.as_array(point_out, shape=(2,)).copy()
    return TangencyResult(float(t_out.value), point, float(mu_out.value))


def pdist_tangency(coef: numpy.ndarray) -> numpy.ndarray:
    lib = _ensure_available()

    func = lib.pdist_tangency
    func.restype = ctypes.c_int
    func.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]

    coef_arr = numpy.ascontiguousarray(coef, dtype=float)
    m = coef_arr.shape[0]
    n = m * (m - 1) // 2
    output = numpy.empty(n, dtype=float)
    error_buffer = ctypes.create_string_buffer(_ERROR_BUFFER)

    status = func(
        coef_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(m),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(error_buffer, ctypes.c_void_p),
        ctypes.c_size_t(_ERROR_BUFFER),
    )

    if status != 0:  # pragma: no cover - propagated to Python layer
        message = error_buffer.value.decode("utf-8", errors="ignore")
        _raise_backend_error(message)

    return output
