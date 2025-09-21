from __future__ import annotations

"""Tangency solver dispatching between Python and C++ backends."""

from typing import Iterable, Tuple, cast, get_args

import numpy

from . import _solver_python as _py
from . import _tangency_cpp as _cpp

__all__ = [
    "quad_eval",
    "pencil",
    "TangencyResult",
    "solve_mu",
    "tangency",
    "pdist_tangency",
    "tangency_python",
    "pdist_tangency_python",
    "has_cpp_backend",
]


quad_eval = _py.quad_eval
pencil = _py.pencil
TangencyResult = _py.TangencyResult
solve_mu = _py.solve_mu

tangency_python = _py.tangency
pdist_tangency_python = _py.pdist_tangency
_pdist_tangency_serial = _py._pdist_tangency_serial
_pdist_tangency_parallel = _py._pdist_tangency_parallel


MethodName = _py.MethodName
_METHOD_NAMES: tuple[str, ...] = tuple(get_args(MethodName))


BackendLiteral = tuple[str, ...]
_BACKEND_NAMES: BackendLiteral = ("auto", "python", "cpp")


def has_cpp_backend() -> bool:
    """Return True if the compiled tangency backend is available."""

    return _cpp.is_available()


def _extract_coef_array(ellcloud: Iterable[numpy.ndarray]) -> numpy.ndarray:
    coef = getattr(ellcloud, "coef", ellcloud)
    array = numpy.asarray(coef, dtype=float)
    if array.ndim != 2 or array.shape[1] != 6:
        raise ValueError("Expected ellipse coefficients with shape (n, 6)")
    return array


def _should_use_cpp(backend: str) -> bool:
    if backend not in _BACKEND_NAMES:
        raise ValueError(
            f"Unknown backend '{backend}'. Expected one of {', '.join(_BACKEND_NAMES)}"
        )
    if backend == "cpp":
        if not has_cpp_backend():
            raise RuntimeError("C++ backend requested but not available")
        return True
    if backend == "auto":
        return has_cpp_backend()
    return False


def _normalize_method(method: MethodName | str) -> MethodName:
    if method not in _METHOD_NAMES:
        raise ValueError(f"Unknown method: {method}")
    return cast(MethodName, method)


def tangency(
    pcoef: numpy.ndarray,
    qcoef: numpy.ndarray,
    *,
    method: MethodName | str = "brentq+newton",
    bracket: Tuple[float, float] = (0.0, 1.0),
    x0: float | None = None,
    backend: str = "auto",
) -> TangencyResult:
    """Return (t, point, μ) at which two ellipses are tangent."""

    method_literal = _normalize_method(method)
    if _should_use_cpp(backend):
        return _cpp.tangency(
            pcoef,
            qcoef,
            method=method_literal,
            bracket=bracket,
            x0=x0,
        )
    return tangency_python(
        pcoef,
        qcoef,
        method=method_literal,
        bracket=bracket,
        x0=x0,
    )


def pdist_tangency(
    ellcloud,
    *,
    parallel: bool = True,
    n_jobs: int | None = -1,
    backend: str = "auto",
) -> numpy.ndarray:
    """Compute pairwise tangency distances for a cloud of ellipses.

    Parameters
    ----------
    ellcloud
        Collection of ellipse coefficient arrays or an `EllipseCloud`.
    parallel : bool, optional
        If True (default), compute the tangencies in parallel when using the
        Python backend.
    n_jobs : int or None, optional
        Number of jobs passed to the Python parallel backend.
    backend : {"auto", "python", "cpp"}
        Backend used for the tangency computation.
    """

    if _should_use_cpp(backend):
        coef = _extract_coef_array(ellcloud)
        return _cpp.pdist_tangency(coef)
    if parallel:
        return _pdist_tangency_parallel(ellcloud, n_jobs=n_jobs)
    return _pdist_tangency_serial(ellcloud)
