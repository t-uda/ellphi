"""Pairwise-aligned many-body tangency API (provisional names).

The names in this module are provisional.  ``tangency_simplex`` extends the
pairwise :func:`ellphi.tangency` contract to one or more packed ellipsoid
coefficient vectors while keeping the many-body numerical engine internal.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from ._minimax_python import MethodName, solve_minimax
from .geometry import unpack_conic

__all__ = [
    "SimplexTangencyResult",
    "SimplexTangencyGrad",
    "tangency_simplex",
    "tangency_simplex_grad",
]


_NEGATIVE_ALPHA_ROUNDING_FACTOR = 64.0
_MACHINE_EPSILON = np.finfo(float).eps


class SimplexTangencyResult(NamedTuple):
    """Result of a many-body tangency calculation.

    ``support`` is the thresholded weight support ``{i: mu[i] > weight_tol}``.
    ``active_set`` is the tight-constraint set at the returned point,
    ``I = {i : t**2 - f_i(point) <= active_tol * max(1, t**2)}``.  Up to
    numerical tolerance, ``support`` is a subset of ``active_set``; under
    strict complementarity (ND1), they coincide.  In the degenerate example
    of unit balls centred at ``(0, 0)``, ``(2, 0)``, and ``(0, 2)``, support
    has two indices while ``active_set`` has three.

    The private engine's internal field named ``active_set`` is the weight
    support (historical ellcech naming) and is not part of the public API.

    Attributes:
        t: Tangency time ``sqrt(max_i f_i(point))``.
        point: Tangency point, shape ``(d,)``.
        mu: Simplex multipliers, shape ``(k,)``.
        support: Indices with multiplier greater than ``weight_tol``.
        active_set: Tight constraints satisfying the numerical active-set
            test with ``active_tol``.
    """

    t: float
    point: np.ndarray
    mu: np.ndarray
    support: tuple[int, ...]
    active_set: tuple[int, ...]


class SimplexTangencyGrad(NamedTuple):
    """Many-body tangency result and coefficient-space gradient.

    ``support`` and ``active_set`` have the definitions documented for
    :class:`SimplexTangencyResult`.

    Attributes:
        t: Tangency time.
        point: Tangency point, shape ``(d,)``.
        mu: Simplex multipliers, shape ``(k,)``.
        dt_dcoef: Gradient of ``t`` for independent packed coefficient
            perturbations, shape ``(k, m)``.
        support: Indices with multiplier greater than ``weight_tol``.
        active_set: Tight constraints at ``point`` under ``active_tol``.
    """

    t: float
    point: np.ndarray
    mu: np.ndarray
    dt_dcoef: np.ndarray
    support: tuple[int, ...]
    active_set: tuple[int, ...]


def _coefficient_basis(point: np.ndarray) -> np.ndarray:
    """Return the packed conic monomial basis at ``point``."""
    tri_i, tri_j = np.triu_indices(point.shape[0])
    quadratic = np.where(
        tri_i == tri_j,
        point[tri_i] ** 2,
        2.0 * point[tri_i] * point[tri_j],
    )
    return np.concatenate((quadratic, 2.0 * point, np.array([1.0])))


def _validate_tolerance(
    name: str, value: float, *, strictly_positive: bool = False
) -> None:
    """Validate a finite numerical tolerance."""
    if not np.isfinite(value) or (value <= 0.0 if strictly_positive else value < 0.0):
        qualifier = "finite and > 0" if strictly_positive else "finite and non-negative"
        raise ValueError(f"{name} must be {qualifier}")


def _validate_positive_int(name: str, value: int) -> None:
    """Validate a positive integer solver parameter."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")


def _validate_solver_parameters(
    k: int,
    *,
    tol: float,
    max_iter: int,
    weight_tol: float,
    regularization: float,
    condition_number_limit: float | None,
    max_conditioning_steps: int,
    newton_tol: float,
    newton_max_iter: int,
) -> None:
    """Validate public solver controls before dispatching to the engine."""
    _validate_tolerance("tol", tol, strictly_positive=True)
    _validate_tolerance("newton_tol", newton_tol)
    _validate_tolerance("regularization", regularization)
    _validate_tolerance("weight_tol", weight_tol)
    _validate_positive_int("max_iter", max_iter)
    _validate_positive_int("newton_max_iter", newton_max_iter)
    _validate_positive_int("max_conditioning_steps", max_conditioning_steps)
    if condition_number_limit is not None and (
        not np.isfinite(condition_number_limit) or condition_number_limit <= 1.0
    ):
        raise ValueError("condition_number_limit must be finite and > 1 when provided")
    if weight_tol >= 1.0 / k:
        raise ValueError("weight_tol must satisfy 0 <= weight_tol < 1/k")


def _prepare_coefs(
    coefs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Reconstruct centers and preserve exact completed-square offsets.

    A packed row encodes the constant as
    ``c = xbar.T @ A @ xbar + delta``. Computing ``delta`` from a
    far-translated or ill-conditioned row can therefore lose precision of
    order ``eps * cond2(A) * max(abs(c), abs(xbar.T @ A @ xbar))``. The
    returned ``value_roundoff`` is the sum of these per-row bounds; callers
    needing exact normalization at large translations should center their data
    first.
    """
    matrices, linear, constants = unpack_conic(coefs)
    centers = np.empty_like(linear)
    offsets = np.empty_like(constants)
    value_roundoff = 0.0
    for row, (matrix, vector, constant) in enumerate(zip(matrices, linear, constants)):
        try:
            inverse_times_linear = np.linalg.solve(matrix, vector)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                f"coefficient row {row} has a singular quadratic matrix"
            ) from exc

        centered_constant = float(vector @ inverse_times_linear)
        delta = float(constant) - centered_constant
        row_roundoff = (
            _NEGATIVE_ALPHA_ROUNDING_FACTOR
            * _MACHINE_EPSILON
            * float(np.linalg.cond(matrix, 2))
            * max(abs(float(constant)), abs(centered_constant))
        )
        if not np.isfinite(delta) or not np.isfinite(row_roundoff):
            raise ValueError(
                f"coefficient row {row} has a non-finite normalization residual"
            )
        centers[row] = -inverse_times_linear
        offsets[row] = delta
        value_roundoff += row_roundoff
    return matrices, centers, offsets, float(value_roundoff)


def _centered_constraint_values(
    point: np.ndarray,
    matrices: np.ndarray,
    centers: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    """Evaluate all centered constraints with the engine's offsets."""
    differences = point[np.newaxis, :] - centers
    values = np.einsum("ki,kij,kj->k", differences, matrices, differences)
    return values + offsets


def tangency_simplex(
    coefs: np.ndarray,
    *,
    method: MethodName | str = "fw+bisect",
    tol: float = 1e-9,
    max_iter: int = 2000,
    weight_tol: float = 1e-10,
    active_tol: float = 1e-9,
    regularization: float = 0.0,
    condition_number_limit: float | None = None,
    max_conditioning_steps: int = 8,
    newton_tol: float = 1e-14,
    newton_max_iter: int = 20,
) -> SimplexTangencyResult:
    """Compute the common tangency time for a simplex of ellipsoids.

    Args:
        coefs: Packed conic coefficient vectors, shape ``(k, m)``.
        method: Internal many-body solver method.
        tol: Frank-Wolfe gap tolerance.
        max_iter: Maximum Frank-Wolfe iterations.
        weight_tol: Threshold defining ``support``.
        active_tol: Relative tolerance defining ``active_set`` as the
            tight-constraint set at the returned point.
        regularization: Optional diagonal regularization of weighted matrices.
        condition_number_limit: Optional weighted-matrix condition target.
        max_conditioning_steps: Maximum conditioning-shift escalations.
        newton_tol: Newton-polishing residual tolerance.
        newton_max_iter: Maximum Newton-polishing iterations.

    Returns:
        A result using tangency time rather than squared filtration value.

    Raises:
        ValueError: If the coefficient array is not two-dimensional, is empty,
            a tolerance is not finite and non-negative, or a coefficient row
            has a singular quadratic matrix, or the packed quadrics have no
            common non-negative tangency scale. Packed input requires
            ``d >= 2``, as for :func:`ellphi.tangency`.
        RuntimeError: If the internal solver does not converge or produces a
            non-finite output.  Solver diagnostics are included in the error.
    """
    coefs = np.asarray(coefs, dtype=float)
    if coefs.ndim != 2:
        raise ValueError("Expected coefficient array with shape (k, m)")
    if coefs.shape[0] == 0:
        raise ValueError("simplex must contain at least one vertex (k=0 given)")
    if coefs.shape[1] == 3:
        raise ValueError(
            "packed input requires d >= 2, as for ellphi.tangency; "
            "one-dimensional packed conics are not supported"
        )
    _validate_tolerance("active_tol", active_tol)
    _validate_solver_parameters(
        coefs.shape[0],
        tol=tol,
        max_iter=max_iter,
        weight_tol=weight_tol,
        regularization=regularization,
        condition_number_limit=condition_number_limit,
        max_conditioning_steps=max_conditioning_steps,
        newton_tol=newton_tol,
        newton_max_iter=newton_max_iter,
    )
    matrices, centers, offsets, value_roundoff = _prepare_coefs(coefs)

    result = solve_minimax(
        matrices,
        centers,
        offsets=offsets,
        method=method,
        tol=tol,
        max_iter=max_iter,
        weight_tol=weight_tol,
        regularization=regularization,
        condition_number_limit=condition_number_limit,
        max_conditioning_steps=max_conditioning_steps,
        newton_tol=newton_tol,
        newton_max_iter=newton_max_iter,
    )
    finite = (
        np.isfinite(result.alpha)
        and np.all(np.isfinite(result.circumcenter))
        and np.all(np.isfinite(result.weights))
    )
    if not result.converged or not finite:
        reason = "did not converge" if not result.converged else "was non-finite"
        raise RuntimeError(
            "tangency_simplex "
            f"{reason}: method={result.method!r}, n_iter={result.n_iter}, "
            f"diagnostics={result.metadata!r}"
        )

    values = _centered_constraint_values(
        result.circumcenter, matrices, centers, offsets
    )
    if not np.all(np.isfinite(values)):
        raise RuntimeError(
            "tangency_simplex was non-finite: "
            f"method={result.method!r}, n_iter={result.n_iter}, "
            f"diagnostics={result.metadata!r}"
        )
    alpha = float(result.alpha)
    constraint_scale = float(np.max(np.abs(values)))
    scale = max(abs(alpha), constraint_scale)
    negative_tolerance = max(
        _NEGATIVE_ALPHA_ROUNDING_FACTOR * _MACHINE_EPSILON * max(1.0, scale),
        value_roundoff,
    )
    if alpha < -negative_tolerance:
        raise ValueError("packed quadrics have no common non-negative tangency scale")
    t_squared = float(max(alpha, 0.0))
    t = float(np.sqrt(max(0.0, t_squared)))
    support = tuple(int(i) for i in np.flatnonzero(result.weights > weight_tol))
    active_threshold = active_tol * max(1.0, t_squared)
    active_set = tuple(
        int(i) for i in np.flatnonzero(t_squared - values <= active_threshold)
    )

    return SimplexTangencyResult(
        t=t,
        point=result.circumcenter,
        mu=result.weights,
        support=support,
        active_set=active_set,
    )


def tangency_simplex_grad(
    coefs: np.ndarray, **solver_kwargs: Any
) -> SimplexTangencyGrad:
    """Return many-body tangency and its packed-coefficient gradient.

    For EllPHi's packed conic ``f_i(x) = coef_i @ basis(x)``, the basis is
    ``[x_j**2 if j == l else 2*x_j*x_l for j <= l, 2*x_0, ..., 2*x_(d-1), 1]``
    in NumPy upper-triangle order.  The envelope theorem gives
    ``d(t**2)/d coef_i = mu_i * basis(point)`` and therefore
    ``dt/d coef_i = mu_i * basis(point) / (2*t)``.

    The engine passes the completed-square offset
    ``delta_i = c_i - b_i^T A_i^-1 b_i`` through the minimax problem. Thus
    ``dt_dcoef`` is the true gradient for independent packed-coordinate
    perturbations, including the constant component ``mu_i / (2*t)``.

    A packed row encodes ``c_i = xbar_i^T A_i xbar_i + delta_i``. For
    ``|xbar_i|**2 >> 1`` or ill-conditioned ``A_i``, the representable
    ``delta_i`` carries roundoff of order
    ``eps * cond2(A_i) * max(|c_i|, |xbar_i^T A_i xbar_i|)``. Callers
    needing exact normalization at large translations should center their data
    first.

    This is a gradient of ``t``, not ``t**2``.  Its hypotheses are a
    non-degenerate input, at least two indices in ``active_set``, multipliers
    supported on ``active_set``, and an exact weighted linear solve. Solver
    convergence or a small residual alone is not a derivative certificate.
    For far-translated inputs, evaluating this coefficient-space derivative in
    the uncentred packed basis can itself be poorly conditioned.

    Args:
        coefs: Packed conic coefficient vectors, shape ``(k, m)``.
        **solver_kwargs: Forwarded to :func:`tangency_simplex`.

    Returns:
        Tangency data and ``dt_dcoef`` with shape ``(k, m)``.

    Raises:
        ZeroDivisionError: If the tangency time is zero.
        RuntimeError: If the forward solve fails.
    """
    coefs = np.asarray(coefs, dtype=float)
    result = tangency_simplex(coefs, **solver_kwargs)
    if result.t == 0.0:
        raise ZeroDivisionError("tangency gradient is undefined when t == 0")

    basis = _coefficient_basis(result.point)
    dt_dcoef = result.mu[:, np.newaxis] * basis[np.newaxis, :] / (2.0 * result.t)
    return SimplexTangencyGrad(
        t=result.t,
        point=result.point,
        mu=result.mu,
        dt_dcoef=dt_dcoef,
        support=result.support,
        active_set=result.active_set,
    )
