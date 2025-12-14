from __future__ import annotations

"""Tangency solver dispatching between Python and C++ backends."""

from typing import Iterable, Tuple, cast, get_args

import numpy

from . import _solver_python as _py
from . import _tangency_cpp as _cpp
from .geometry import infer_dim_from_coef_length

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
    "MethodName",
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
    """Checks if the compiled C++ backend for tangency calculations is available.

    Returns:
        `True` if the C++ backend is available, `False` otherwise.
    """
    return _cpp.is_available()


def _extract_coef_array(ellcloud: Iterable[numpy.ndarray]) -> numpy.ndarray:
    coef = getattr(ellcloud, "coef", ellcloud)
    array = numpy.asarray(coef, dtype=float)
    if array.ndim == 3 and array.shape[1] == 1:
        array = array[:, 0, :]
    if array.ndim != 2:
        raise ValueError("Expected coefficient array with shape (m, n)")
    infer_dim_from_coef_length(array.shape[1])
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


def _resolve_backend_hybrid_iterations(
    dim: int,
    hybrid_bracket_maxiter: int | None,
    hybrid_newton_maxiter: int | None,
) -> tuple[int, int]:
    return _py._resolve_hybrid_iterations(
        dim, hybrid_bracket_maxiter, hybrid_newton_maxiter
    )


def tangency(
    pcoef: numpy.ndarray,
    qcoef: numpy.ndarray,
    *,
    method: MethodName | str = "brentq+newton",
    bracket: Tuple[float, float] = (0.0, 1.0),
    x0: float | None = None,
    backend: str = "auto",
    hybrid_bracket_maxiter: int | None = None,
    hybrid_newton_maxiter: int | None = None,
    failsafe: bool = True,
) -> TangencyResult:
    """Computes the tangency point between two ellipses.

    This function returns the tangency time `t`, the tangent point `point`,
    and the pencil parameter `μ` at which the two ellipses are tangent.

    Args:
        pcoef: The coefficient vector for the first ellipse.
        qcoef: The coefficient vector for the second ellipse.
        method: The root-finding method to use. Can be one of "brentq+newton",
            "brentq", "brenth", "bisect", or "newton".
        bracket: The bracketing interval for bracket methods.
        x0: An optional initial guess for Newton's method. Required if
            `method` is "newton".
        backend: The backend to use for the computation. Can be one of "auto",
            "cpp", or "python".
        hybrid_bracket_maxiter: An optional maximum number of iterations for
            the bracket phase in the hybrid method.
        hybrid_newton_maxiter: An optional maximum number of iterations for
            the Newton phase in the hybrid method.
        failsafe: If `True`, a failsafe fallback is enabled. If Newton
            refinement fails to converge in the hybrid method, the function
            falls back to the high-precision Brent's method.

    Returns:
        A `TangencyResult` named tuple with the following fields:
        - `t`: The tangency time.
        - `point`: The tangent point.
        - `mu`: The pencil parameter `μ` at which the two ellipses are tangent.
    """
    method_literal = _normalize_method(method)
    pcoef_arr = numpy.asarray(pcoef, dtype=float).reshape(-1)
    qcoef_arr = numpy.asarray(qcoef, dtype=float).reshape(-1)
    if pcoef_arr.shape != qcoef_arr.shape:
        raise ValueError("Coefficient vectors must have the same length")
    coef_length = pcoef_arr.size
    infer_dim_from_coef_length(coef_length)

    dim = infer_dim_from_coef_length(coef_length)
    if method_literal == "brentq+newton":
        bracket_iter, newton_iter = _resolve_backend_hybrid_iterations(
            dim,
            hybrid_bracket_maxiter,
            hybrid_newton_maxiter,
        )
        if bracket_iter <= 0:
            raise ValueError("hybrid_bracket_maxiter must be positive")
        if newton_iter <= 0:
            raise ValueError("hybrid_newton_maxiter must be positive")
    else:
        default_bracket, default_newton = _py._hybrid_iteration_defaults(dim)
        bracket_iter = (
            default_bracket
            if hybrid_bracket_maxiter is None
            else hybrid_bracket_maxiter
        )
        newton_iter = (
            default_newton if hybrid_newton_maxiter is None else hybrid_newton_maxiter
        )

    if backend not in _BACKEND_NAMES:
        raise ValueError(
            f"Unknown backend '{backend}'. Expected one of {', '.join(_BACKEND_NAMES)}"
        )
    use_cpp = backend in {"cpp", "auto"} and _should_use_cpp(backend)

    if use_cpp:
        return _cpp.tangency(
            pcoef_arr,
            qcoef_arr,
            method=method_literal,
            bracket=bracket,
            x0=x0,
            hybrid_bracket_maxiter=bracket_iter,
            hybrid_newton_maxiter=newton_iter,
            failsafe=failsafe,
        )
    return tangency_python(
        pcoef_arr,
        qcoef_arr,
        method=method_literal,
        bracket=bracket,
        x0=x0,
        hybrid_bracket_maxiter=bracket_iter,
        hybrid_newton_maxiter=newton_iter,
        failsafe=failsafe,
    )


def pdist_tangency(
    ellcloud,
    *,
    parallel: bool = True,
    n_jobs: int | None = -1,
    backend: str = "auto",
) -> numpy.ndarray:
    """Computes pairwise tangency distances for a cloud of ellipses.

    Args:
        ellcloud: A collection of ellipse coefficient arrays or an
            `EllipseCloud` object.
        parallel: If `True`, the computation is performed in parallel when
            using the Python backend.
        n_jobs: The number of jobs to run in parallel.
        backend: The backend to use for the computation. Can be one of
            "auto", "python", or "cpp".

    Returns:
        A condensed distance matrix of tangency distances.
    """
    if backend not in _BACKEND_NAMES:
        raise ValueError(
            f"Unknown backend '{backend}'. Expected one of {', '.join(_BACKEND_NAMES)}"
        )

    if backend in {"cpp", "auto"}:
        coef = _extract_coef_array(ellcloud)
        if _should_use_cpp(backend):
            return _cpp.pdist_tangency(coef)

    if parallel:
        return _pdist_tangency_parallel(ellcloud, n_jobs=n_jobs)
    return _pdist_tangency_serial(ellcloud)
