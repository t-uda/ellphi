from __future__ import annotations

"""Tangency solver – consolidated version with correct derivative formula."""

from collections import namedtuple
from typing import Tuple

import numpy
from scipy.optimize import root_scalar
from typing import Callable, Literal, cast

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ellphi.ellcloud import EllipseCloud
# from .ellcloud import EllipseCloud

__all__ = [
    "quad_eval",
    "pencil",
    "TangencyResult",
    "tangency",
    "pdist_tangency",
]


# ---------------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------------


def quad_eval(coef: numpy.ndarray, center: Tuple[float, float]) -> float:
    """Evaluate quadratic form *ax² + 2bxy + cy² + 2dx + 2ey + f*"""
    assert coef.shape == (6,)
    a, b, c, d, e, f = coef[:6]
    x, y = center
    return a * x**2 + 2 * b * x * y + c * y**2 + 2 * d * x + 2 * e * y + f


def pencil(p: numpy.ndarray, q: numpy.ndarray, mu: float) -> numpy.ndarray:
    """Linear blend ``(1-μ) p + μ q`` of two conic-coefficient arrays."""
    return (1.0 - mu) * p + mu * q


# ---------------------------------------------------------------------------
# Tangency solver internals
# ---------------------------------------------------------------------------

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
    coef = pencil(p, q, mu)
    diff = p - q  # derivative of pencil wrt mu is (q-p); sign handled below

    # Centre of blended ellipse
    xc = _center(coef)

    # Build 2×2 matrices
    A_mu = numpy.array([[coef[0], coef[1]], [coef[1], coef[2]]])
    diff_mat = numpy.array([[diff[0], diff[1]], [diff[1], diff[2]]])

    # A_xprime = -(diff_mat @ xc + diff[3:5])
    A_xprime = -(diff_mat @ xc + diff[3:5])

    # Inverse of A_mu
    A_mu_inv = numpy.linalg.inv(A_mu)

    # 2 * A_xprimeᵀ · A_mu_inv · A_xprime
    return float(2.0 * (A_xprime.T @ A_mu_inv @ A_xprime))


def _solve_mu(
    p: numpy.ndarray,
    q: numpy.ndarray,
    *,
    method: str = "brentq+newton",
    bracket: Tuple[float, float] = (0.0, 1.0),
    x0: float | None = None,
) -> float:
    curry_f: Callable[[float], float] = lambda mu: _target(mu, p, q)
    curry_df: Callable[[float], float] = lambda mu: _target_prime(mu, p, q)
    if method == "brentq+newton":
        mu0 = root_scalar(curry_f, bracket=bracket, method="brentq", maxiter=8).root
        mu = root_scalar(
            curry_f,
            x0=mu0,
            method="newton",
            fprime=curry_df,
            maxiter=3,
        ).root
        return float(mu)
    if method in {"bisect", "brentq", "brenth"}:
        return float(
            root_scalar(
                curry_f,
                bracket=bracket,
                method=cast(Literal["bisect", "brentq", "brenth"], method),
            ).root
        )
    if method == "newton":
        if x0 is None:
            raise ValueError("x0 must be provided for Newton method")
        return float(root_scalar(curry_f, x0=x0, method="newton", fprime=curry_df).root)
    raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def tangency(
    pcoef: numpy.ndarray,
    qcoef: numpy.ndarray,
    *,
    method: str = "brentq+newton",
    bracket: Tuple[float, float] = (0.0, 1.0),
    x0: float | None = None,
) -> TangencyResult:
    """Return (t, point, μ) at which two ellipses are tangent."""
    mu = _solve_mu(pcoef, qcoef, method=method, bracket=bracket, x0=x0)
    coef = pencil(pcoef, qcoef, mu)
    point = _center(coef)
    t = float(numpy.sqrt(quad_eval(coef, point)))
    return TangencyResult(t, numpy.asarray(point), mu)


def pdist_tangency(ellcloud: EllipseCloud) -> numpy.ndarray:
    """
    The pairwise tangency is computed and stored in entry
    ``m * i + j - ((i + 2) * (i + 1)) // 2``,
    where m is the number of ellipses.
    """
    m = len(ellcloud)
    n = m * (m - 1) // 2
    d = numpy.zeros((n,), dtype=float)
    for i in range(m):
        for j in range(i + 1, m):
            k = m * i + j - ((i + 2) * (i + 1)) // 2
            d[k] = tangency(ellcloud[i], ellcloud[j]).t
    return d
