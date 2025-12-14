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
from scipy import linalg
from scipy.optimize import root_scalar, newton as scipy_newton

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


# --- Algebraic Sigmoid Transform Helpers ---


def _x_from_u(u: float | numpy.ndarray) -> float | numpy.ndarray:
    """x(u) = 0.5 * (1 + u / sqrt(1 + u^2)), maps R -> (0, 1)"""
    return 0.5 * (1.0 + u / numpy.sqrt(1.0 + u**2))


def _u_from_x(x: float | numpy.ndarray) -> float | numpy.ndarray:
    """Inverse: u = (2x - 1) / (2 * sqrt(x(1-x)))"""
    is_scalar = not isinstance(x, numpy.ndarray)
    x_arr = numpy.atleast_1d(x)

    # Initialize output array with NaNs for out-of-domain values
    u = numpy.full(x_arr.shape, numpy.nan)

    # Handle boundary cases: x=0 -> u=-inf, x=1 -> u=inf
    u[x_arr == 0] = -numpy.inf
    u[x_arr == 1] = numpy.inf

    # Process valid interior points (0 < x < 1)
    mask = (x_arr > 0) & (x_arr < 1)
    x_safe = x_arr[mask]

    # Add a small epsilon to the denominator to prevent division by zero
    # for values extremely close to 0 or 1.
    denominator = 2.0 * numpy.sqrt(x_safe * (1.0 - x_safe) + 1e-30)
    u[mask] = (2.0 * x_safe - 1.0) / denominator

    return u.item() if is_scalar else u


def _x_prime_from_u(u: float | numpy.ndarray) -> float | numpy.ndarray:
    """x'(u) = dx/du = 0.5 * (1 + u^2)^(-1.5)"""
    return 0.5 * (1.0 + u**2) ** (-1.5)


# --- Linear Solvers ---


def _gaussian_elimination(matrix: numpy.ndarray, rhs: numpy.ndarray) -> numpy.ndarray:
    """Solve ``Ax = rhs`` with partial pivoting.

    This mirrors the C++ fallback implementation so that the Python backend
    behaves consistently across NumPy releases when Cholesky factorisation
    fails.
    """

    A = numpy.array(matrix, dtype=float, copy=True)
    b = numpy.array(rhs, dtype=float, copy=True)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise numpy.linalg.LinAlgError("Matrix must be square")

    dim = A.shape[0]
    for k in range(dim):
        pivot = k + int(numpy.argmax(numpy.abs(A[k:, k])))
        pivot_value = A[pivot, k]
        if pivot_value == 0.0:
            raise numpy.linalg.LinAlgError("Matrix is singular")

        if pivot != k:
            A[[k, pivot]] = A[[pivot, k]]
            b[[k, pivot]] = b[[pivot, k]]

        diag = A[k, k]
        factors = A[k + 1 :, k] / diag
        b[k + 1 :] -= factors * b[k]
        A[k + 1 :, k:] -= factors[:, numpy.newaxis] * A[k : k + 1, k:]

    x = numpy.zeros(dim, dtype=float)
    for i in range(dim - 1, -1, -1):
        diag = A[i, i]
        if diag == 0.0:
            raise numpy.linalg.LinAlgError("Matrix is singular")
        residual = b[i] - A[i, i + 1 :] @ x[i + 1 :]
        x[i] = residual / diag

    return x


# --- Core Solver Functions ---


def quad_eval(coef: numpy.ndarray, center: Tuple[float, ...] | numpy.ndarray) -> float:
    """Evaluates the quadratic form `xᵀAx + 2bᵀx + c` for the given coefficients.

    Args:
        coef: The coefficient vector of the quadratic form.
        center: The point at which to evaluate the quadratic form.

    Returns:
        The value of the quadratic form at the given point.
    """
    A, b, c = unpack_single_conic(coef)
    x = numpy.asarray(center, dtype=float)
    if x.ndim != 1 or x.shape[0] != b.shape[0]:
        raise ValueError("Point dimensionality does not match conic coefficients")
    return float(x @ A @ x + 2.0 * b @ x + c)


def pencil(p: numpy.ndarray, q: numpy.ndarray, mu: float) -> numpy.ndarray:
    """Computes the linear blend `(1-μ) p + μ q` of two conic-coefficient arrays.

    Args:
        p: The first conic-coefficient array.
        q: The second conic-coefficient array.
        mu: The blending factor.

    Returns:
        The blended conic-coefficient array.
    """
    return (1.0 - mu) * p + mu * q


TangencyResult = namedtuple("TangencyResult", ["t", "point", "mu"])
NewtonResult = namedtuple("NewtonResult", ["root", "converged"])


def _algsig_newton_py(
    curry_f: Callable[[float], float],
    curry_df: Callable[[float], float],
    x0: float,
    maxiter: int,
    xtol: float,
    rtol: float,
) -> NewtonResult:
    """Algebraic-sigmoid transformed Newton-Raphson method (Python implementation)."""
    u = 0.0 if x0 is None else _u_from_x(x0)
    if not numpy.isfinite(u):
        u = 0.0  # Fallback for invalid x0

    for i in range(maxiter):
        x = _x_from_u(u)
        f_val = curry_f(x)

        if not numpy.isfinite(f_val):
            return NewtonResult(x, False)

        f_prime_val = curry_df(x)
        x_prime_val = _x_prime_from_u(u)
        F_prime_u = f_prime_val * x_prime_val

        if not numpy.isfinite(F_prime_u) or F_prime_u == 0:
            return NewtonResult(x, False)

        # Backtracking Line Search
        delta_u = -f_val / F_prime_u
        alpha = 1.0
        u_next = u
        step_accepted = False

        for j in range(10):  # Max 10 backtracking steps
            u_candidate = u + alpha * delta_u
            if not numpy.isfinite(u_candidate):
                alpha *= 0.5
                continue

            f_candidate = curry_f(_x_from_u(u_candidate))

            # Relaxed check: allow equality to handle cases where we hit machine
            # precision and cannot strictly improve, effectively falling through to
            # convergence check.
            if numpy.isfinite(f_candidate) and abs(f_candidate) <= abs(f_val):
                u_next = u_candidate
                step_accepted = True
                break
            alpha *= 0.5

        if not step_accepted:
            return NewtonResult(x, False)  # Backtracking failed

        # Convergence criterion
        if abs(u_next - u) <= xtol + rtol * abs(u_next):
            return NewtonResult(_x_from_u(u_next), True)

        u = u_next

    return NewtonResult(_x_from_u(u), False)


def _center(coef: numpy.ndarray) -> numpy.ndarray:
    A, b, _ = unpack_single_conic(coef)
    try:
        chol = linalg.cho_factor(A, check_finite=False)
        center = linalg.cho_solve(chol, -b, check_finite=False)
    except linalg.LinAlgError:
        try:
            center = _gaussian_elimination(A, -b)
        except numpy.linalg.LinAlgError:  # pragma: no cover - defensive
            # If the fallback also fails, propagate NaN
            return numpy.full(A.shape[0], numpy.nan, dtype=float)
    return center


def _target(mu: float, p: numpy.ndarray, q: numpy.ndarray) -> float:
    coef = pencil(p, q, mu)
    xc = _center(coef)
    return quad_eval(p, xc) - quad_eval(q, xc)


def _target_prime(mu: float, p: numpy.ndarray, q: numpy.ndarray) -> float:
    """Exact derivative of `_target`."""

    try:
        pencil = build_tangent_pencil(mu, p, q)
    except (numpy.linalg.LinAlgError, linalg.LinAlgError, ZeroDivisionError):
        return float("nan")
    return target_prime_from_pencil(pencil, p, q)


SingleStageMethodName = Literal["bisect", "brentq", "brenth", "newton"]
MethodName = Literal[
    "brentq+newton", "algsig+newton", "bisect", "brentq", "brenth", "newton"
]
_BRACKET_METHODS: tuple[SingleStageMethodName, ...] = ("bisect", "brentq", "brenth")
_DEFAULT_HYBRID_BRACKET_MAXITER = 28
_DEFAULT_HYBRID_NEWTON_MAXITER = 3
_HYBRID_BRACKET_MAXITER_FAILSAFE = 64
_NEWTON_ONLY_MAXITER = 50
_NEWTON_RTOL = 4.0 * numpy.finfo(float).eps
_NEWTON_XTOL = 1e-8


def _hybrid_iteration_defaults(dim: int) -> tuple[int, int]:
    # The dimension argument `dim` is preserved for API compatibility, even
    # though the defaults are now dimension-independent.
    return (_DEFAULT_HYBRID_BRACKET_MAXITER, _DEFAULT_HYBRID_NEWTON_MAXITER)


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


def _initial_mu_for_newton(curry_df: Callable[[float], float]) -> float:
    """Choose an interior start biased toward larger |F'| near the endpoints."""

    eps = 1e-5
    candidates = [eps, 1.0 - eps]
    scores: list[float] = []
    for mu in candidates:
        try:
            scores.append(abs(curry_df(mu)))
        except Exception:
            scores.append(float("nan"))

    if numpy.all(numpy.isnan(scores)):
        return 0.5
    idx = int(numpy.nanargmax(numpy.asarray(scores)))
    return float(candidates[idx])


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
    failsafe: bool = True,
) -> float:
    """Solves for the pencil parameter `μ` at which two ellipses are tangent.

    Args:
        p: The coefficient vector for the first ellipse.
        q: The coefficient vector for the second ellipse.
        method: The root-finding method to use.
        bracket: The bracketing interval for bracket methods.
        x0: An optional initial guess for Newton's method.
        hybrid_bracket_maxiter: An optional maximum number of iterations for
            the bracket phase in the hybrid method.
        hybrid_newton_maxiter: An optional maximum number of iterations for
            the Newton phase in the hybrid method.
        failsafe: If `True`, a failsafe fallback is enabled.

    Returns:
        The pencil parameter `μ` at which the two ellipses are tangent.
    """
    curry_f = cast(Callable[[float], float], partial(_target, p=p, q=q))
    curry_df = cast(Callable[[float], float], partial(_target_prime, p=p, q=q))

    def _ensure_finite(
        func: Callable[[float], float], label: str
    ) -> Callable[[float], float]:
        def wrapper(mu: float) -> float:
            value = func(mu)
            if not numpy.isfinite(value):
                msg = f"Non-finite {label} value during Newton iteration"
                raise RuntimeError(msg)
            return value

        return wrapper

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

        try:
            # SciPy's Newton uses a similar tolerance scheme, so we can rely on it
            root, result = scipy_newton(
                _ensure_finite(curry_f, "target"),
                x0=mu0,
                fprime=_ensure_finite(curry_df, "target derivative"),
                maxiter=newton_iter,
                full_output=True,
                disp=False,
                tol=_NEWTON_XTOL,
                rtol=_NEWTON_RTOL,
            )

            if result.converged:
                return float(root)
        except (RuntimeError, OverflowError):
            # If Newton diverges catastrophically, failsafe will handle it
            pass

        if failsafe:
            return solve_single_stage(
                "brentq",
                bracket=bracket,
                maxiter=_HYBRID_BRACKET_MAXITER_FAILSAFE,
            )
        return mu0

    if method == "algsig+newton":
        mu_guess = 0.5 if x0 is None else x0
        result = _algsig_newton_py(
            curry_f,
            curry_df,
            mu_guess,
            _NEWTON_ONLY_MAXITER,
            _NEWTON_XTOL,
            _NEWTON_RTOL,
        )
        if result.converged:
            return result.root

        if failsafe:
            return solve_single_stage(
                "brentq",
                bracket=bracket,
                maxiter=_HYBRID_BRACKET_MAXITER_FAILSAFE,
            )

        # If not converged & no failsafe, raise error to match scipy.newton's behavior
        raise RuntimeError("algsig+newton failed to converge.")

    if method in _BRACKET_METHODS:
        return solve_single_stage(cast(SingleStageMethodName, method), bracket=bracket)

    if method == "newton":
        if x0 is None:
            raise ValueError("x0 must be provided for Newton method")

        try:
            root, result = scipy_newton(
                _ensure_finite(curry_f, "target"),
                x0=x0,
                fprime=_ensure_finite(curry_df, "target derivative"),
                maxiter=_NEWTON_ONLY_MAXITER,
                full_output=True,
                disp=False,  # Important: disp=False raises RuntimeError on failure
                tol=_NEWTON_XTOL,
                rtol=_NEWTON_RTOL,
            )
            if result.converged:
                return float(root)
        except (RuntimeError, OverflowError):
            # Let failsafe handle it, otherwise re-raise
            if not failsafe:
                raise

        if failsafe:
            return solve_single_stage(
                "brentq",
                bracket=bracket,
                maxiter=_HYBRID_BRACKET_MAXITER_FAILSAFE,
            )
        # This part should ideally not be reached if disp=False, but as a fallback:
        raise RuntimeError("Newton method failed to converge.")

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
    failsafe: bool = True,
) -> TangencyResult:
    """Computes the tangency point between two ellipses using the Python backend.

    Args:
        pcoef: The coefficient vector for the first ellipse.
        qcoef: The coefficient vector for the second ellipse.
        method: The root-finding method to use.
        bracket: The bracketing interval for bracket methods.
        x0: An optional initial guess for Newton's method.
        hybrid_bracket_maxiter: An optional maximum number of iterations for
            the bracket phase in the hybrid method.
        hybrid_newton_maxiter: An optional maximum number of iterations for
            the Newton phase in the hybrid method.
        failsafe: If `True`, a failsafe fallback is enabled.

    Returns:
        A `TangencyResult` named tuple with the following fields:
        - `t`: The tangency time.
        - `point`: The tangent point.
        - `mu`: The pencil parameter `μ` at which the two ellipses are tangent.

    Raises:
        RuntimeError: If the solver fails to converge to a valid solution
            within the bracket.
    """
    mu = solve_mu(
        pcoef,
        qcoef,
        method=method,
        bracket=bracket,
        x0=x0,
        hybrid_bracket_maxiter=hybrid_bracket_maxiter,
        hybrid_newton_maxiter=hybrid_newton_maxiter,
        failsafe=failsafe,
    )
    if not (numpy.isfinite(mu) and bracket[0] <= mu <= bracket[1]):
        raise RuntimeError("Solver failed to find a root within the bracket")

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
    """Computes pairwise tangency distances for a cloud of ellipses.

    Args:
        ellcloud: An `EllipseCloud` object containing the ellipses.
        parallel: If `True`, the computation is performed in parallel.
        n_jobs: The number of jobs to run in parallel.

    Returns:
        A condensed distance matrix of tangency distances.
    """
    if parallel:
        return _pdist_tangency_parallel(ellcloud, n_jobs=n_jobs)
    return _pdist_tangency_serial(ellcloud)
