"""Anisotropic Čech filtration API.

The :func:`cech` function computes the anisotropic Čech filtration time for
one or more packed ellipsoid coefficient vectors while keeping the many-body
numerical engine internal.  For ``k = 2`` the Čech time equals the tangency
time computed by :func:`ellphi.tangency`, and ``cech`` agrees with it.
"""

from __future__ import annotations

from typing import Literal, NamedTuple, cast

import numpy as np

from ._minimax_python import MethodName as EngineMethodName
from ._minimax_python import MinimaxResult, solve_minimax
from .geometry import unpack_conic

__all__ = [
    "MethodName",
    "CechStage",
    "CechInfo",
    "CechResult",
    "CechGrad",
    "cech",
    "cech_grad",
]


MethodName = Literal[
    "fw+bisect",
    "fw+brentq",
    "fw+bisect+newton",
    "fw+brentq+newton",
    "fw+bisect+damped-newton",
    "scipy-slsqp",
    "newton-cold",
    "auto",
]

_NAMED_METHODS = (
    "fw+bisect",
    "fw+brentq",
    "fw+bisect+newton",
    "fw+brentq+newton",
    "fw+bisect+damped-newton",
    "scipy-slsqp",
    "newton-cold",
    "fw+newton",
)


_NEGATIVE_ALPHA_ROUNDING_FACTOR = 64.0
_MACHINE_EPSILON = np.finfo(float).eps


class CechStage(NamedTuple):
    """Diagnostics for one public Čech solver stage."""

    method: str
    converged: bool
    status: str
    gap: float
    n_iter: int


class CechInfo(NamedTuple):
    """Traceability information for a public Čech solve.

    ``n_iter`` is the iteration or evaluation count reported by the selected
    stage.  The counts and outcomes for every attempted stage are in
    ``stages``.
    """

    requested_method: str
    method_used: str
    converged: bool
    gap: float
    n_iter: int
    stages: tuple[CechStage, ...]


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
    ``I = {i : t**2 - f_i(point) <= active_tol * max(1, t**2)}``; when
    ``alpha`` is clipped at zero, the actual shift ``t**2 - alpha`` is added
    to this threshold.  Up to numerical tolerance, ``support`` is a subset of
    ``active_set``; under strict complementarity (ND1), they coincide.  In the
    degenerate example
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
        info: Requested method, selected method, final gap and per-stage
            convergence diagnostics.
    """

    t: float
    point: np.ndarray
    mu: np.ndarray
    support: tuple[int, ...]
    active_set: tuple[int, ...]
    info: CechInfo


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
        info: Requested method, selected method, final gap and per-stage
            convergence diagnostics.
    """

    t: float
    point: np.ndarray
    mu: np.ndarray
    dt_dcoef: np.ndarray
    support: tuple[int, ...]
    active_set: tuple[int, ...]
    info: CechInfo


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


def _validate_quadratic_matrices(matrices: np.ndarray) -> np.ndarray:
    """Validate symmetry and positive definiteness before solving centers."""
    lambda_minima = np.empty(matrices.shape[0], dtype=float)
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
            eigenvalues = np.linalg.eigvalsh(matrix)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                f"coefficient row {row} has a non-positive definite quadratic matrix"
            ) from exc
        lambda_min = float(eigenvalues[0])
        if not np.isfinite(lambda_min) or lambda_min <= 0.0:
            raise ValueError(
                f"coefficient row {row} has a non-positive definite quadratic matrix"
            )
        lambda_minima[row] = lambda_min
    return lambda_minima


def _prepare_coefs(
    coefs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct centers and estimate completed-square roundoff.

    A packed row encodes the constant as
    ``c = xbar.T @ A @ xbar + delta``. Computing ``delta`` from a
    far-translated row can therefore lose precision in the centered quadratic
    term and its subtraction from ``c``. The returned per-row rounding
    estimates use the computed solve residual and the dot-product/subtraction
    dimension; the minimax value is a convex combination of row offsets.
    Callers needing exact normalization at large translations should center
    their data first.
    """
    matrices, linear, constants = unpack_conic(coefs)
    lambda_minima = _validate_quadratic_matrices(matrices)
    centers = np.empty_like(linear)
    offsets = np.empty_like(constants)
    row_roundoff_estimates = np.empty_like(constants)
    for row, (matrix, vector, constant) in enumerate(zip(matrices, linear, constants)):
        try:
            y_hat = np.linalg.solve(matrix, vector)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                f"coefficient row {row} has a singular quadratic matrix"
            ) from exc

        residual = vector - matrix @ y_hat
        centered_constant = float(vector @ y_hat)
        delta = float(constant) - centered_constant
        solve_estimate = float(
            np.linalg.norm(vector, ord=2)
            * np.linalg.norm(residual, ord=2)
            / lambda_minima[row]
        )
        rounding_estimate = (
            (matrix.shape[0] + 2)
            * _MACHINE_EPSILON
            * (abs(float(constant)) + abs(centered_constant))
        )
        row_roundoff = solve_estimate + rounding_estimate
        if not np.isfinite(delta) or not np.isfinite(row_roundoff):
            raise ValueError(
                f"coefficient row {row} has a non-finite normalization residual"
            )
        centers[row] = -y_hat
        offsets[row] = delta
        row_roundoff_estimates[row] = row_roundoff
    return matrices, centers, offsets, row_roundoff_estimates


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


class _StageRun(NamedTuple):
    result: MinimaxResult | None
    stage: CechStage
    values: np.ndarray | None
    dual_value: float
    gap_scale: float
    error: str | None


def _stage_status(
    result: MinimaxResult,
    *,
    accepted: bool,
    finite: bool,
    finite_values: bool,
) -> str:
    """Return a stable public status for one engine result."""
    if accepted:
        return "converged"
    if not finite or not finite_values:
        return "nonfinite"
    if result.metadata is not None:
        newton_status = result.metadata.get("newton_status")
        if isinstance(newton_status, str) and newton_status != "converged":
            return newton_status
    return "gap_not_met"


def _run_stage(
    matrices: np.ndarray,
    centers: np.ndarray,
    offsets: np.ndarray,
    *,
    method: str,
    tol: float,
    max_iter: int,
    weight_tol: float,
    newton_tol: float,
    newton_max_iter: int,
) -> _StageRun:
    """Run one engine stage and retain its public convergence diagnostics."""
    try:
        result = solve_minimax(
            matrices,
            centers,
            offsets=offsets,
            method=cast(EngineMethodName, method),
            tol=tol,
            max_iter=max_iter,
            weight_tol=weight_tol,
            newton_tol=newton_tol,
            newton_max_iter=newton_max_iter,
        )
    except Exception as exc:
        stage = CechStage(
            method=method,
            converged=False,
            status=f"exception:{type(exc).__name__}",
            gap=float("nan"),
            n_iter=0,
        )
        return _StageRun(None, stage, None, float("nan"), float("nan"), str(exc))

    finite = bool(
        np.isfinite(result.alpha)
        and np.all(np.isfinite(result.circumcenter))
        and np.all(np.isfinite(result.weights))
    )
    values = _centered_constraint_values(
        result.circumcenter, matrices, centers, offsets
    )
    finite_values = bool(np.all(np.isfinite(values)))
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
    accepted = bool(
        result.converged
        and finite
        and finite_values
        and np.isfinite(final_gap)
        and final_gap <= tol * gap_scale
    )
    status = _stage_status(
        result,
        accepted=accepted,
        finite=finite,
        finite_values=finite_values,
    )
    stage = CechStage(
        method=result.method,
        converged=accepted,
        status=status,
        gap=final_gap,
        n_iter=result.n_iter,
    )
    return _StageRun(result, stage, values, dual_value, gap_scale, None)


def _stage_diagnostics(run: _StageRun) -> str:
    """Format one stage for a failure message."""
    details = (
        f"method={run.stage.method!r}, status={run.stage.status!r}, "
        f"final_gap={run.stage.gap:.17g}, n_iter={run.stage.n_iter}"
    )
    if run.result is not None and run.result.metadata is not None:
        n_fevals = run.result.metadata.get("n_fevals")
        if n_fevals is not None:
            details += f", n_fevals={n_fevals}"
    if run.error:
        details += f", cause={run.error!r}"
    return details


def _raise_named_failure(run: _StageRun, tol: float) -> None:
    """Raise the legacy-shaped diagnostic for a named method."""
    raise RuntimeError(
        "cech did not converge: "
        f"method={run.stage.method!r}, status={run.stage.status!r}, "
        f"iterations={run.stage.n_iter}, duality_gap={run.stage.gap:.17g}, "
        f"tol={tol:g}, duality_gap_scale={run.gap_scale:.17g}; "
        f"{_stage_diagnostics(run)}; "
        "a larger max_iter or a different method may be chosen"
    )


def _raise_auto_failure(runs: tuple[_StageRun, ...]) -> None:
    """Raise a diagnostic retaining both failed auto-route stages."""
    details = "; ".join(
        f"stage {index}: {_stage_diagnostics(run)}"
        for index, run in enumerate(runs, start=1)
    )
    raise RuntimeError(f"cech auto route failed: {details}")


def _assemble_result(
    run: _StageRun,
    *,
    requested_method: str,
    stages: tuple[CechStage, ...],
    row_roundoff_estimates: np.ndarray,
    weight_tol: float,
    active_tol: float,
) -> CechResult:
    """Build the public result from an accepted stage."""
    assert run.result is not None
    assert run.values is not None
    result = run.result
    values = run.values
    alpha = float(result.alpha)
    support_indices = np.flatnonzero(result.weights > weight_tol)
    support = tuple(int(i) for i in support_indices)
    support_scale = float(np.max(np.abs(values[support_indices])))
    scale = max(abs(alpha), support_scale)
    positive_indices = np.flatnonzero(result.weights > 0.0)
    # The accepted allowance is weighted over every row with a positive multiplier;
    # the public support remains thresholded by weight_tol.
    roundoff_estimate = float(
        np.dot(
            result.weights[positive_indices],
            row_roundoff_estimates[positive_indices],
        )
    )
    negative_tolerance = max(
        _NEGATIVE_ALPHA_ROUNDING_FACTOR * _MACHINE_EPSILON * max(1.0, scale),
        roundoff_estimate,
    )
    if alpha < -negative_tolerance:
        raise ValueError("packed quadrics have no common non-negative filtration scale")
    t_squared = float(max(alpha, 0.0))
    t = float(np.sqrt(max(0.0, t_squared)))
    clipping_shift = t_squared - alpha
    active_threshold = active_tol * max(1.0, t_squared) + clipping_shift
    active_set = tuple(
        int(i) for i in np.flatnonzero(t_squared - values <= active_threshold)
    )
    info = CechInfo(
        requested_method=requested_method,
        method_used=run.stage.method,
        converged=run.stage.converged,
        gap=run.stage.gap,
        n_iter=run.stage.n_iter,
        stages=stages,
    )
    return CechResult(
        t=t,
        point=result.circumcenter,
        mu=result.weights,
        support=support,
        active_set=active_set,
        info=info,
    )


def cech(
    coefs: np.ndarray,
    *,
    method: MethodName | str = "auto",
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
        method: Internal many-body solver method. The default ``"auto"``
            first runs ``"fw+brentq+newton"`` with the caller's ``tol``,
            ``max_iter``, and ``newton_tol``. If that stage does not converge,
            it runs gap-enforced ``"scipy-slsqp"`` from uniform dual weights
            with the same ``tol``. A named method never falls back.
        tol: Pairwise Frank-Wolfe gap tolerance, relative to max(1, |dual
            value|).
        max_iter: Maximum Frank-Wolfe iterations.
        weight_tol: Threshold defining the public ``support``; it does not
            remove positive-weight rows from the normalization estimate.
        active_tol: Relative tolerance for the public ``active_set`` tightness
            test, with only the actual zero-clipping shift added when clipping
            occurs.
        newton_tol: Residual tolerance for Newton's own polishing iterations;
            it does not alter the shared accepted-result gap test.
        newton_max_iter: Maximum Newton-polishing iterations.

    Stabilized solves are available only through the private minimax engine;
    they are research controls and carry no public gradient guarantee.

    Returns:
        A result containing the Čech filtration time rather than the squared
        filtration value. The trailing ``info`` field records the requested
        method, selected method, final gap, count, and every attempted stage.

    Raises:
        ValueError: If the coefficient array is not two-dimensional, is empty,
            a tolerance is not finite and non-negative, or a coefficient row
            has a non-symmetric or non-positive-definite quadratic matrix, or
            the packed quadrics have no common non-negative filtration scale.
            Packed input requires
            ``d >= 2``, as for :func:`ellphi.tangency`.
        RuntimeError: If a named method does not converge or produces a
            non-finite output, or if both stages of the ``"auto"`` route fail.
            The error reports stage statuses, final gaps, and iteration or
            evaluation counts.
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
    matrices, centers, offsets, row_roundoff_estimates = _prepare_coefs(coefs)

    if method != "auto" and method not in _NAMED_METHODS:
        raise ValueError(f"Unknown method {method!r}")
    requested_method = str(method)
    stage_methods = (
        ("fw+brentq+newton", "scipy-slsqp") if method == "auto" else (requested_method,)
    )
    runs: list[_StageRun] = []
    for stage_method in stage_methods:
        run = _run_stage(
            matrices,
            centers,
            offsets,
            method=stage_method,
            tol=tol,
            max_iter=max_iter,
            weight_tol=weight_tol,
            newton_tol=newton_tol,
            newton_max_iter=newton_max_iter,
        )
        runs.append(run)
        if run.stage.converged:
            return _assemble_result(
                run,
                requested_method=requested_method,
                stages=tuple(item.stage for item in runs),
                row_roundoff_estimates=row_roundoff_estimates,
                weight_tol=weight_tol,
                active_tol=active_tol,
            )
        if method != "auto":
            _raise_named_failure(run, tol)
    _raise_auto_failure(tuple(runs))
    raise AssertionError("unreachable")


def cech_grad(
    coefs: np.ndarray,
    *,
    method: MethodName | str = "auto",
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
        method: Internal many-body solver method. The default ``"auto"``
            first runs ``"fw+brentq+newton"`` with the caller's ``tol``,
            ``max_iter``, and ``newton_tol``. If that stage does not converge,
            it runs gap-enforced ``"scipy-slsqp"`` from uniform dual weights
            with the same ``tol``. A named method never falls back.
        tol: Pairwise Frank-Wolfe gap tolerance, relative to max(1, |dual
            value|).
        max_iter: Maximum Frank-Wolfe iterations.
        weight_tol: Threshold defining the public ``support``; it does not
            remove positive-weight rows from the normalization estimate.
        active_tol: Relative tolerance for the public ``active_set`` tightness
            test, with only the actual zero-clipping shift added when clipping
            occurs.
        newton_tol: Residual tolerance for Newton's own polishing iterations;
            it does not alter the shared accepted-result gap test.
        newton_max_iter: Maximum Newton-polishing iterations.

    Returns:
        Čech filtration data and ``dt_dcoef`` with shape ``(k, m)``. The
        trailing ``info`` field mirrors the forward solve's diagnostics.

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
        info=result.info,
    )
