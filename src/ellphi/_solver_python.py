from __future__ import annotations

"""Pure Python tangency solver backend."""

from collections import namedtuple
from functools import partial
from itertools import combinations
from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal, Tuple, cast

import numpy
from joblib import Parallel, delayed  # type: ignore
from scipy.optimize import root_scalar

from ._tangent_pencil import build_tangent_pencil, target_prime_from_pencil

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


def quad_eval(coef: numpy.ndarray, center: Tuple[float, float]) -> float:
    """Evaluate quadratic form *ax² + 2bxy + cy² + 2dx + 2ey + f*."""

    assert coef.shape == (6,)
    a, b, c, d, e, f = coef[:6]
    x, y = center
    return a * x**2 + 2 * b * x * y + c * y**2 + 2 * d * x + 2 * e * y + f


def pencil(p: numpy.ndarray, q: numpy.ndarray, mu: float) -> numpy.ndarray:
    """Linear blend ``(1-μ) p + μ q`` of two conic-coefficient arrays."""

    return (1.0 - mu) * p + mu * q


TangencyResult = namedtuple("TangencyResult", ["t", "point", "mu"])


def _center(coef: numpy.ndarray) -> Tuple[float, float]:
    a, b, c, d, e, _ = coef
    det = a * c - b**2
    if det == 0:
        raise ZeroDivisionError("Degenerate conic (determinant zero)")
    x = (b * e - c * d) / det
    y = (b * d - a * e) / det
    return (x, y)


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


def solve_mu(
    p: numpy.ndarray,
    q: numpy.ndarray,
    *,
    method: MethodName = "brentq+newton",
    bracket: Tuple[float, float] = (0.0, 1.0),
    x0: float | None = None,
) -> float:
    curry_f = cast(Callable[[float], float], partial(_target, p=p, q=q))
    curry_df = cast(Callable[[float], float], partial(_target_prime, p=p, q=q))

    def solve_single_stage(method_name: SingleStageMethodName, **kwargs: Any) -> float:
        if method_name == "newton":
            kwargs.setdefault("fprime", curry_df)
        result = root_scalar(curry_f, method=method_name, **kwargs)
        return float(result.root)

    if method == "brentq+newton":
        mu0 = solve_single_stage("brentq", bracket=bracket, maxiter=8)
        return solve_single_stage("newton", x0=mu0, maxiter=3)
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
) -> TangencyResult:
    """Return (t, point, μ) at which two ellipses are tangent."""

    mu = solve_mu(pcoef, qcoef, method=method, bracket=bracket, x0=x0)
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

    results = Parallel(n_jobs=n_jobs)(
        delayed(get_pair_tangency)(i, j) for _, (i, j) in pairs
    )
    return numpy.asarray(results, dtype=float)


def pdist_tangency(
    ellcloud: EllipseCloud, *, parallel: bool = True, n_jobs: int | None = -1
) -> numpy.ndarray:
    if parallel:
        return _pdist_tangency_parallel(ellcloud, n_jobs=n_jobs)
    return _pdist_tangency_serial(ellcloud)
