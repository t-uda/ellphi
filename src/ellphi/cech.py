"""Anisotropic Čech filtration API.

The :func:`cech` function computes the anisotropic Čech filtration time for
one or more packed ellipsoid coefficient vectors while keeping the many-body
numerical engine internal.  For ``k = 2`` the Čech time equals the tangency
time computed by :func:`ellphi.tangency`, and ``cech`` agrees with it.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ._minimax_python import MethodName, solve_minimax
from .geometry import unpack_conic

__all__ = [
    "CechResult",
    "CechGrad",
    "cech",
    "cech_grad",
]


_NEGATIVE_ALPHA_ROUNDING_FACTOR = 64.0
_MACHINE_EPSILON = np.finfo(float).eps


class CechResult(NamedTuple):
    """Result of an anisotropic Čech filtration calculation.

    For a simplex ``sigma`` of ellipsoids with quadratic functions ``f_i``,
    ``t**2 = alpha(sigma) = min_x max_i f_i(x)`` is the least squared scale at
    which all growing ellipsoids ``E_i(t) = {f_i <= t**2}`` share a point.
    At the critical scale their intersection is the single point ``x*``;
    active boundaries pass through ``x*`` and satisfy
    ``sum_i mu_i grad f_i(x*) = 0`` with positive weights, but the boundaries
    are not tangent to each other.  Pairwise tangency is only the ``k = 2``
    special case.

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
        t: Čech filtration time; ``t**2 = alpha(sigma)``.
        point: Common intersection point ``x*`` at the critical scale, shape
            ``(d,)``.
        mu: Lagrange multipliers (dual weights on the simplex), shape ``(k,)``.
        support: Indices with multiplier greater than ``weight_tol``.
        active_set: Tight constraints satisfying the numerical active-set
            test with ``active_tol``.
    """

    t: float
    point: np.ndarray
    mu: np.ndarray
    support: tuple[int, ...]
    active_set: tuple[int, ...]


class CechGrad(NamedTuple):
    """Čech filtration result and coefficient-space gradient.

    ``support`` and ``active_set`` have the definitions documented for
    :class:`CechResult`.

    Attributes:
        t: Čech filtration time; ``t**2 = alpha(sigma)``.
        point: Common intersection point ``x*`` at the critical scale, shape
            ``(d,)``.
        mu: Lagrange multipliers (dual weights on the simplex), shape ``(k,)``.
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
    newton_tol: float,
    newton_max_iter: int,
) -> None:
    """Validate public solver controls before dispatching to the engine."""
    _validate_tolerance("tol", tol, strictly_positive=True)
    _validate_tolerance("newton_tol", newton_tol, strictly_positive=True)
    _validate_tolerance("weight_tol", weight_tol)
    _validate_positive_int("max_iter", max_iter)
    _validate_positive_int("newton_max_iter", newton_max_iter)
    if weight_tol >= 1.0 / k:
        raise ValueError("weight_tol must satisfy 0 <= weight_tol < 1/k")


def _validate_quadratic_matrices(matrices: np.ndarray) -> None:
    """Validate symmetry and positive definiteness before solving centers."""
    for row, matrix in enumerate(matrices):
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"coefficient row {row} has a non-finite quadratic matrix")

        condition_number = float(np.linalg.cond(matrix, 2))
        if not np.isfinite(condition_number):
            condition_number = 1.0
        symmetry_rtol = (
            _NEGATIVE_ALPHA_ROUNDING_FACTOR
            * _MACHINE_EPSILON
            * max(1.0, condition_number)
        )
        matrix_scale = float(np.linalg.norm(matrix, ord=np.inf))
        symmetry_error = float(np.linalg.norm(matrix - matrix.T, ord=np.inf))
        if symmetry_error > symmetry_rtol * matrix_scale:
            raise ValueError(
                f"coefficient row {row} has a non-symmetric quadratic matrix"
            )

        try:
            np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                f"coefficient row {row} has a non-positive definite quadratic matrix"
            ) from exc


def _prepare_coefs(
    coefs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct centers and preserve exact completed-square offsets.

    A packed row encodes the constant as
    ``c = xbar.T @ A @ xbar + delta``. Computing ``delta`` from a
    far-translated row can therefore lose precision in the centered quadratic
    term and its subtraction from ``c``. The returned per-row bounds are based
    on those computed terms and the actual linear solve; the minimax value is a
    convex combination of row offsets.
    Callers needing exact normalization at large translations should center
    their data first.
    """
    matrices, linear, constants = unpack_conic(coefs)
    _validate_quadratic_matrices(matrices)
    centers = np.empty_like(linear)
    offsets = np.empty_like(constants)
    row_roundoff_bounds = np.empty_like(constants)
    for row, (matrix, vector, constant) in enumerate(zip(matrices, linear, constants)):
        try:
            inverse_times_linear = np.linalg.solve(matrix, vector)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                f"coefficient row {row} has a singular quadratic matrix"
            ) from exc

        centered_constant = float(vector @ inverse_times_linear)
        delta = float(constant) - centered_constant
        solve_scale = float(
            np.linalg.norm(vector, ord=2) * np.linalg.norm(inverse_times_linear, ord=2)
        )
        row_roundoff = _NEGATIVE_ALPHA_ROUNDING_FACTOR * _MACHINE_EPSILON * float(
            np.linalg.cond(matrix, 2)
        ) * solve_scale + _NEGATIVE_ALPHA_ROUNDING_FACTOR * _MACHINE_EPSILON * (
            abs(float(constant)) + abs(centered_constant)
        )
        if not np.isfinite(delta) or not np.isfinite(row_roundoff):
            raise ValueError(
                f"coefficient row {row} has a non-finite normalization residual"
            )
        centers[row] = -inverse_times_linear
        offsets[row] = delta
        row_roundoff_bounds[row] = row_roundoff
    return matrices, centers, offsets, row_roundoff_bounds


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


def cech(
    coefs: np.ndarray,
    *,
    method: MethodName | str = "fw+brentq+newton",
    tol: float = 1e-9,
    max_iter: int = 2000,
    weight_tol: float = 1e-10,
    active_tol: float = 1e-9,
    newton_tol: float = 1e-14,
    newton_max_iter: int = 20,
) -> CechResult:
    """Compute the Čech filtration time for a simplex of ellipsoids.

    The returned ``t`` satisfies ``t**2 = alpha(sigma)``, where
    ``alpha(sigma) = min_x max_i f_i(x)`` is the least squared scale at which
    all growing ellipsoids share a common point.  For ``k = 2`` the Čech time
    equals the tangency time computed by :func:`ellphi.tangency`, and ``cech``
    agrees with it.

    Args:
        coefs: Packed conic coefficient vectors, shape ``(k, m)``.
        method: Internal many-body solver method. The default is
            ``"fw+brentq+newton"``, pairwise Frank-Wolfe with adaptive Brent
            line search followed by Newton polishing.
        tol: Pairwise Frank-Wolfe gap tolerance, relative to max(1, |dual
            value|).
        max_iter: Maximum Frank-Wolfe iterations.
        weight_tol: Threshold defining ``support``.
        active_tol: Relative tolerance defining ``active_set`` as the
            tight-constraint set at the returned point.
        newton_tol: Newton-polishing residual tolerance.
        newton_max_iter: Maximum Newton-polishing iterations.

    Stabilized solves are available only through the private minimax engine;
    they are research controls and carry no public gradient guarantee.

    Returns:
        A result containing the Čech filtration time rather than the squared
        filtration value.

    Raises:
        ValueError: If the coefficient array is not two-dimensional, is empty,
            a tolerance is not finite and non-negative, or a coefficient row
            has a non-symmetric or non-positive-definite quadratic matrix, or
            the packed quadrics have no common non-negative filtration scale.
            Packed input requires
            ``d >= 2``, as for :func:`ellphi.tangency`.
        RuntimeError: If the internal solver does not converge or produces a
            non-finite output. The error names the method, iterations, final
            duality gap, and ``tol`` and gives retry guidance.
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
        newton_tol=newton_tol,
        newton_max_iter=newton_max_iter,
    )
    matrices, centers, offsets, row_roundoff_bounds = _prepare_coefs(coefs)

    result = solve_minimax(
        matrices,
        centers,
        offsets=offsets,
        method=method,
        tol=tol,
        max_iter=max_iter,
        weight_tol=weight_tol,
        newton_tol=newton_tol,
        newton_max_iter=newton_max_iter,
    )
    finite = (
        np.isfinite(result.alpha)
        and np.all(np.isfinite(result.circumcenter))
        and np.all(np.isfinite(result.weights))
    )
    values = _centered_constraint_values(
        result.circumcenter, matrices, centers, offsets
    )
    finite_values = np.all(np.isfinite(values))
    final_gap = (
        float(np.max(values) - np.dot(result.weights, values))
        if finite and finite_values
        else float("nan")
    )
    dual_value = (
        float(np.dot(result.weights, values))
        if finite and finite_values
        else float("nan")
    )
    gap_scale = max(1.0, abs(dual_value)) if np.isfinite(dual_value) else float("nan")
    if not result.converged or not finite or not finite_values:
        reason = "did not converge" if not result.converged else "was non-finite"
        raise RuntimeError(
            "cech "
            f"{reason}: method={result.method!r}, iterations={result.n_iter}, "
            f"duality_gap={final_gap:.17g}, tol={tol:g}, "
            f"duality_gap_scale={gap_scale:.17g}; "
            "a larger max_iter or a different method may be chosen"
        )
    alpha = float(result.alpha)
    support_indices = np.flatnonzero(result.weights > weight_tol)
    support = tuple(int(i) for i in support_indices)
    support_scale = float(np.max(np.abs(values[support_indices])))
    scale = max(abs(alpha), support_scale)
    # The accepted allowance is the mu-weighted sum of per-row bounds over support.
    support_roundoff = float(
        np.dot(result.weights[support_indices], row_roundoff_bounds[support_indices])
    )
    negative_tolerance = max(
        _NEGATIVE_ALPHA_ROUNDING_FACTOR * _MACHINE_EPSILON * max(1.0, scale),
        support_roundoff,
    )
    if alpha < -negative_tolerance:
        raise ValueError("packed quadrics have no common non-negative filtration scale")
    t_squared = float(max(alpha, 0.0))
    t = float(np.sqrt(max(0.0, t_squared)))
    active_threshold = active_tol * max(1.0, t_squared) + negative_tolerance
    active_set = tuple(
        int(i) for i in np.flatnonzero(t_squared - values <= active_threshold)
    )

    return CechResult(
        t=t,
        point=result.circumcenter,
        mu=result.weights,
        support=support,
        active_set=active_set,
    )


def cech_grad(
    coefs: np.ndarray,
    *,
    method: MethodName | str = "fw+brentq+newton",
    tol: float = 1e-9,
    max_iter: int = 2000,
    weight_tol: float = 1e-10,
    active_tol: float = 1e-9,
    newton_tol: float = 1e-14,
    newton_max_iter: int = 20,
) -> CechGrad:
    """Return the Čech filtration time and its coefficient-space gradient.

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
    ``delta_i`` carries roundoff from the computed centered quadratic term and
    its subtraction from ``c_i``. Callers needing exact normalization at large
    translations should center their data first.

    This is a gradient of ``t``, not ``t**2``.  Its hypotheses are a
    non-degenerate input, at least two indices in ``active_set``, multipliers
    supported on ``active_set``, and an exact weighted linear solve. Solver
    convergence or a small residual alone is not a derivative certificate.
    For far-translated inputs, evaluating this coefficient-space derivative in
    the uncentred packed basis can itself be poorly conditioned.

    Args:
        coefs: Packed conic coefficient vectors, shape ``(k, m)``.
        method: Internal many-body solver method. The default is
            ``"fw+brentq+newton"``, pairwise Frank-Wolfe with adaptive Brent
            line search followed by Newton polishing.
        tol: Pairwise Frank-Wolfe gap tolerance, relative to max(1, |dual
            value|).
        max_iter: Maximum Frank-Wolfe iterations.
        weight_tol: Threshold defining ``support``.
        active_tol: Relative tolerance defining ``active_set``.
        newton_tol: Newton-polishing residual tolerance.
        newton_max_iter: Maximum Newton-polishing iterations.

    Returns:
        Čech filtration data and ``dt_dcoef`` with shape ``(k, m)``.

    Raises:
        ValueError: If a quadratic block is not symmetric positive definite.
        ZeroDivisionError: If the Čech filtration time is zero.
        RuntimeError: If the forward solve fails.
    """
    coefs = np.asarray(coefs, dtype=float)
    result = cech(
        coefs,
        method=method,
        tol=tol,
        max_iter=max_iter,
        weight_tol=weight_tol,
        active_tol=active_tol,
        newton_tol=newton_tol,
        newton_max_iter=newton_max_iter,
    )
    if result.t == 0.0:
        raise ZeroDivisionError("Čech gradient is undefined when t == 0")

    basis = _coefficient_basis(result.point)
    dt_dcoef = result.mu[:, np.newaxis] * basis[np.newaxis, :] / (2.0 * result.t)
    return CechGrad(
        t=result.t,
        point=result.point,
        mu=result.mu,
        dt_dcoef=dt_dcoef,
        support=result.support,
        active_set=result.active_set,
    )
