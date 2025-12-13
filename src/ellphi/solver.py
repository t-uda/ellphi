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
    """Return True if the compiled tangency backend is available."""

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
    """Compute the tangency point between two ellipses.

    Returns the tangency time `t`, the tangent point `point`, and the
    pencil parameter `μ` at which the two ellipses are tangent.

    Parameters
    ----------
    pcoef : numpy.ndarray
        Coefficient vector for the first ellipse.
    qcoef : numpy.ndarray
        Coefficient vector for the second ellipse.
    method : str, default="brentq+newton"
        Root-finding method. Options: "brentq+newton", "brentq", "brenth",
        "bisect", or "newton".
    bracket : tuple of float, default=(0.0, 1.0)
        Bracketing interval for bracket methods.
    x0 : float, optional
        Initial guess for Newton's method (required if method="newton").
    backend : str, default="auto"
        Backend to use: "auto", "cpp", or "python".
    hybrid_bracket_maxiter : int, optional
        Maximum iterations for the bracket phase in the hybrid method. Defaults
        depend on dimensionality: 28 iterations in 2D and 28 iterations for
        dimensions greater than 2. Explicit values override these defaults as
        resolved by `_resolve_backend_hybrid_iterations`.
    hybrid_newton_maxiter : int, optional
        Maximum iterations for the Newton phase in the hybrid method. Defaults
        depend on dimensionality: 3 iterations in 2D and 3 iterations for
        dimensions greater than 2. Explicit values override these defaults as
        resolved by `_resolve_backend_hybrid_iterations`.
    failsafe : bool, default=True
        Enable failsafe fallback. When True, if Newton refinement fails to
        converge in the hybrid method, falls back to high-precision Brent's
        method (64 iterations). When False, returns the initial bracket result
        if Newton fails, allowing measurement of accuracy degradation.

    Returns
    -------
    TangencyResult
        Named tuple with fields (t, point, mu).
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
