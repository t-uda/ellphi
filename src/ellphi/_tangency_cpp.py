from __future__ import annotations

"""Bindings for the pre-built C++ tangency backend."""

import ctypes
from importlib.metadata import PackageNotFoundError, version as package_version
import sys
import sysconfig
from pathlib import Path
from typing import Tuple

import numpy

from ._solver_python import TangencyResult
from ._version import __version__
from .geometry import infer_dim_from_coef_length

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


def _expected_backend_version() -> str:
    if __version__ != "0+unknown":
        return __version__

    try:
        return package_version(__package__ or "ellphi")
    except PackageNotFoundError:
        return "0+unknown"


def _library_version(lib: ctypes.CDLL) -> str:
    try:
        func = lib.tangency_backend_version
    except AttributeError as exc:
        raise RuntimeError(
            "C++ backend is missing version metadata. Please rebuild the extension."
        ) from exc
    func.restype = ctypes.c_char_p
    value = func()
    if value is None:
        raise RuntimeError("C++ backend returned an empty version string")
    return value.decode("utf-8", errors="replace")


def _validate_library_version(lib: ctypes.CDLL) -> None:
    expected = _expected_backend_version()
    actual = _library_version(lib)
    if actual != expected:
        raise RuntimeError(
            "C++ backend version mismatch: "
            f"library reports '{actual}' but ellphi expects '{expected}'. "
            "Rebuild the extension yourself (e.g., run 'python "
            "build_tangency_cpp.py' or 'poetry install'); it is not rebuilt "
            "automatically during import."
        )


def _load_library() -> ctypes.CDLL:
    lib_path = _library_path()
    if not lib_path.exists():
        raise FileNotFoundError(f"C++ backend library missing: {lib_path}")
    lib = ctypes.CDLL(str(lib_path))
    _validate_library_version(lib)
    return lib


_LIB: ctypes.CDLL | None
_LIB_ERROR: str | None
try:  # pragma: no cover - build process is environment dependent
    _LIB = _load_library()
    _LIB_ERROR = None
except (OSError, FileNotFoundError, RuntimeError) as exc:  # pragma: no cover
    _LIB = None
    _LIB_ERROR = str(exc)


def is_available() -> bool:
    """Checks if the C++ backend is available.

    Returns:
        `True` if the C++ backend is available, `False` otherwise.
    """
    return _LIB is not None


def _ensure_available() -> ctypes.CDLL:
    if _LIB is None:
        message = (
            _LIB_ERROR
            or "C++ backend not available. Run 'python build_tangency_cpp.py' "
            "to build it."
        )
        raise RuntimeError(message)
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
    hybrid_bracket_maxiter: int,
    hybrid_newton_maxiter: int,
    failsafe: bool,
) -> TangencyResult:
    """Computes the tangency point between two ellipses using the C++ backend.

    Args:
        pcoef: The coefficient vector for the first ellipse.
        qcoef: The coefficient vector for the second ellipse.
        method: The root-finding method to use.
        bracket: The bracketing interval for bracket methods.
        x0: An optional initial guess for Newton's method.
        hybrid_bracket_maxiter: The maximum number of iterations for the
            bracket phase in the hybrid method.
        hybrid_newton_maxiter: The maximum number of iterations for the
            Newton phase in the hybrid method.
        failsafe: If `True`, a failsafe fallback is enabled.

    Returns:
        A `TangencyResult` named tuple with the following fields:
        - `t`: The tangency time.
        - `point`: The tangent point.
        - `mu`: The pencil parameter `μ` at which the two ellipses are tangent.
    """
    lib = _ensure_available()

    func = lib.tangency_solve
    func.restype = ctypes.c_int
    func.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]

    pcoef_arr = numpy.ascontiguousarray(pcoef, dtype=float)
    qcoef_arr = numpy.ascontiguousarray(qcoef, dtype=float)
    bracket_arr = numpy.ascontiguousarray(bracket, dtype=float)
    if pcoef_arr.ndim != 1 or qcoef_arr.ndim != 1:
        raise ValueError("Coefficient vectors must be one-dimensional")
    if pcoef_arr.shape != qcoef_arr.shape:
        raise ValueError("Coefficient vectors must have the same length")
    coef_length = pcoef_arr.shape[0]
    dim = infer_dim_from_coef_length(coef_length)

    t_out = ctypes.c_double()
    point_out = (ctypes.c_double * dim)()
    mu_out = ctypes.c_double()
    error_buffer = ctypes.create_string_buffer(_ERROR_BUFFER)

    has_x0 = 0 if x0 is None else 1
    x0_val = 0.0 if x0 is None else float(x0)

    status = func(
        pcoef_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        qcoef_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(coef_length),
        method.encode("utf-8"),
        bracket_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(has_x0),
        ctypes.c_double(x0_val),
        ctypes.c_int(int(hybrid_bracket_maxiter)),
        ctypes.c_int(int(hybrid_newton_maxiter)),
        ctypes.c_int(1 if failsafe else 0),
        ctypes.byref(t_out),
        point_out,
        ctypes.c_size_t(dim),
        ctypes.byref(mu_out),
        ctypes.cast(error_buffer, ctypes.c_void_p),
        ctypes.c_size_t(_ERROR_BUFFER),
    )

    if status != 0:  # pragma: no cover - propagated to Python layer
        message = error_buffer.value.decode("utf-8", errors="ignore")
        _raise_backend_error(message)

    point = numpy.ctypeslib.as_array(point_out, shape=(dim,)).copy()
    return TangencyResult(float(t_out.value), point, float(mu_out.value))


def pdist_tangency(coef: numpy.ndarray) -> numpy.ndarray:
    """Computes pairwise tangency distances for a cloud of ellipses.

    Args:
        coef: A NumPy array of shape `(m, n)` containing the coefficient
            vectors for `m` ellipses.

    Returns:
        A condensed distance matrix of tangency distances.
    """
    lib = _ensure_available()

    func = lib.pdist_tangency
    func.restype = ctypes.c_int
    func.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]

    coef_arr = numpy.ascontiguousarray(coef, dtype=float)
    if coef_arr.ndim != 2:
        raise ValueError("Coefficient array must have shape (m, n)")
    coef_length = coef_arr.shape[1]
    infer_dim_from_coef_length(coef_length)
    m = coef_arr.shape[0]
    n = m * (m - 1) // 2
    output = numpy.empty(n, dtype=float)
    error_buffer = ctypes.create_string_buffer(_ERROR_BUFFER)

    status = func(
        coef_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(m),
        ctypes.c_size_t(coef_length),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(error_buffer, ctypes.c_void_p),
        ctypes.c_size_t(_ERROR_BUFFER),
    )

    if status != 0:  # pragma: no cover - propagated to Python layer
        message = error_buffer.value.decode("utf-8", errors="ignore")
        _raise_backend_error(message)

    return output
