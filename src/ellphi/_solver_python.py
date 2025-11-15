from __future__ import annotations

"""Pure Python tangency solver backend."""

from collections import namedtuple
from functools import partial
from itertools import combinations
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    Literal,
    Tuple,
    cast,
)

import numpy
from joblib import Parallel, delayed  # type: ignore
from scipy.optimize import root_scalar

from ._tangent_pencil import build_tangent_pencil, target_prime_from_pencil
from .geometry import infer_dim_from_coef_length, unpack_single_conic

if TYPE_CHECKING:  # pragma: no cover - only for typing
    from ellphi.ellcloud import EllipseCloud

__all__ = [
    "quad_eval",
    "pencil",
    "TangencyResult",
    "solve_mu",
    "tangency",
    "_pdist_tangency_serial",
    "_pdist_tangency_parallel",
]


def quad_eval(coef: numpy.ndarray, center: Tuple[float, ...] | numpy.ndarray) -> float:
    """Evaluate ``xᵀAx + 2bᵀx + c`` for the provided coefficients."""

    A, b, c = unpack_single_conic(coef)
    x = numpy.asarray(center, dtype=float)
    if x.ndim != 1 or x.shape[0] != b.shape[0]:
        raise ValueError("Point dimensionality does not match conic coefficients")
    return float(x @ A @ x + 2.0 * b @ x + c)


def pencil(p: numpy.ndarray, q: numpy.ndarray, mu: float) -> numpy.ndarray:
    """Linear blend ``(1-μ) p + μ q`` of two conic-coefficient arrays."""

    return (1.0 - mu) * p + mu * q


TangencyResult = namedtuple("TangencyResult", ["t", "point", "mu"])


def _center(coef: numpy.ndarray) -> numpy.ndarray:
    A, b, _ = unpack_single_conic(coef)
    try:
        center = numpy.linalg.solve(A, -b)
    except numpy.linalg.LinAlgError as exc:  # pragma: no cover - defensive
        raise ZeroDivisionError("Degenerate conic (singular quadratic form)") from exc
    return center


def _target(mu: float, p: numpy.ndarray, q: numpy.ndarray) -> float:
    coef = pencil(p, q, mu)
    xc = _center(coef)
    return quad_eval(p, xc) - quad_eval(q, xc)


def _target_prime(mu: float, p: numpy.ndarray, q: numpy.ndarray) -> float:
    """Exact derivative of `_target`."""

    pencil = build_tangent_pencil(mu, p, q)
    return target_prime_from_pencil(pencil, p, q)


SingleStageMethodName = Literal["bisect", "brentq", "brenth", "newton"]
MethodName = Literal["brentq+newton", "bisect", "brentq", "brenth", "newton"]
_BRACKET_METHODS: tuple[SingleStageMethodName, ...] = ("bisect", "brentq", "brenth")
_DEFAULT_HYBRID_BRACKET_MAXITER_2D = 8
_DEFAULT_HYBRID_NEWTON_MAXITER_2D = 3
_DEFAULT_HYBRID_BRACKET_MAXITER_ND = 28
_DEFAULT_HYBRID_NEWTON_MAXITER_ND = 6


def _hybrid_iteration_defaults(dim: int) -> tuple[int, int]:
    if dim == 2:
        return (_DEFAULT_HYBRID_BRACKET_MAXITER_2D, _DEFAULT_HYBRID_NEWTON_MAXITER_2D)
    return (_DEFAULT_HYBRID_BRACKET_MAXITER_ND, _DEFAULT_HYBRID_NEWTON_MAXITER_ND)


def _resolve_hybrid_iterations(
    dim: int,
    hybrid_bracket_maxiter: int | None,
    hybrid_newton_maxiter: int | None,
) -> tuple[int, int]:
    default_bracket, default_newton = _hybrid_iteration_defaults(dim)
    bracket_iter = (
        default_bracket if hybrid_bracket_maxiter is None else hybrid_bracket_maxiter
    )
    newton_iter = (
        default_newton if hybrid_newton_maxiter is None else hybrid_newton_maxiter
    )
    return bracket_iter, newton_iter


def _infer_dim_from_coef(p: numpy.ndarray) -> int:
    coef = numpy.asarray(p, dtype=float).reshape(-1)
    return infer_dim_from_coef_length(coef.size)


def solve_mu(
    p: numpy.ndarray,
    q: numpy.ndarray,
    *,
    method: MethodName = "brentq+newton",
    bracket: Tuple[float, float] = (0.0, 1.0),
    x0: float | None = None,
    hybrid_bracket_maxiter: int | None = None,
    hybrid_newton_maxiter: int | None = None,
) -> float:
    curry_f = cast(Callable[[float], float], partial(_target, p=p, q=q))
    curry_df = cast(Callable[[float], float], partial(_target_prime, p=p, q=q))

    def solve_single_stage(method_name: SingleStageMethodName, **kwargs: Any) -> float:
        if method_name == "newton":
            kwargs.setdefault("fprime", curry_df)
        result = root_scalar(curry_f, method=method_name, **kwargs)
        return float(result.root)

    if method == "brentq+newton":
        dim = _infer_dim_from_coef(p)
        bracket_iter, newton_iter = _resolve_hybrid_iterations(
            dim, hybrid_bracket_maxiter, hybrid_newton_maxiter
        )
        if bracket_iter <= 0:
            raise ValueError("hybrid_bracket_maxiter must be positive")
        if newton_iter <= 0:
            raise ValueError("hybrid_newton_maxiter must be positive")
        mu0 = solve_single_stage("brentq", bracket=bracket, maxiter=bracket_iter)
        return solve_single_stage("newton", x0=mu0, maxiter=newton_iter)
    if method in _BRACKET_METHODS:
        return solve_single_stage(cast(SingleStageMethodName, method), bracket=bracket)
    if method == "newton":
        if x0 is None:
            raise ValueError("x0 must be provided for Newton method")
        return solve_single_stage("newton", x0=x0)
    raise ValueError(f"Unknown method: {method}")


def tangency(
    pcoef: numpy.ndarray,
    qcoef: numpy.ndarray,
    *,
    method: MethodName = "brentq+newton",
    bracket: Tuple[float, float] = (0.0, 1.0),
    x0: float | None = None,
    hybrid_bracket_maxiter: int | None = None,
    hybrid_newton_maxiter: int | None = None,
) -> TangencyResult:
    """Return (t, point, μ) at which two ellipses are tangent."""

    mu = solve_mu(
        pcoef,
        qcoef,
        method=method,
        bracket=bracket,
        x0=x0,
        hybrid_bracket_maxiter=hybrid_bracket_maxiter,
        hybrid_newton_maxiter=hybrid_newton_maxiter,
    )
    coef = pencil(pcoef, qcoef, mu)
    point = _center(coef)
    t = float(numpy.sqrt(quad_eval(coef, point)))
    return TangencyResult(t, numpy.asarray(point), mu)


def _indexed_pairs(size: int) -> Iterator[tuple[int, tuple[int, int]]]:
    """Return ordered ellipse index pairs with their position."""

    return enumerate(combinations(range(size), 2))


def _pdist_tangency_serial(ellcloud: EllipseCloud) -> numpy.ndarray:
    """Serial implementation of pdist_tangency."""

    m = len(ellcloud)
    n = m * (m - 1) // 2
    d = numpy.zeros((n,), dtype=float)
    for k, (i, j) in _indexed_pairs(m):
        d[k] = tangency(ellcloud[i], ellcloud[j]).t
    return d


def _pdist_tangency_parallel(
    ellcloud: EllipseCloud, n_jobs: int | None = -1
) -> numpy.ndarray:
    """Parallel implementation of pdist_tangency."""

    m = len(ellcloud)
    n = m * (m - 1) // 2
    if n == 0:
        return numpy.zeros((0,), dtype=float)

    pairs = _indexed_pairs(m)

    def get_pair_tangency(i: int, j: int) -> float:
        return tangency(ellcloud[i], ellcloud[j]).t

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(get_pair_tangency)(i, j) for _, (i, j) in pairs
    )
    return numpy.asarray(results, dtype=float)


def pdist_tangency(
    ellcloud: EllipseCloud, *, parallel: bool = True, n_jobs: int | None = -1
) -> numpy.ndarray:
    if parallel:
        return _pdist_tangency_parallel(ellcloud, n_jobs=n_jobs)
    return _pdist_tangency_serial(ellcloud)
