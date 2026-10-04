"""Many-body minimax solver for anisotropic filtration values (provisional).

Provenance
----------
Adapted from uda-lab/ellcech ``src/ellphi_alpha/minimax.py`` at commit
82d13e3e174f4903cdcd6adcee1f469348361c91, by Tomoki Uda (the author of
EllPHi). That file is byte-identical at ellcech ``main``
893056409db56865b7432274562db2d863e0abce. This engine deliberately deviates
from ellcech 82d13e3 in the following ways:

* Armijo-rejected Newton steps are never applied. The best iterate by dual
  value is retained, the status is ``"armijo_rejected"``, and
  ``converged=False``.
* After an accepted warm-start Newton step, the weight-thresholded face is
  re-identified before the next step; the cold-start baseline keeps its full
  initial face during Newton.
* Undamped Newton polishing stops when the reduced Hessian is ill-conditioned
  beyond the configured limit, with status ``"ill_conditioned_hessian"``.
  Damped Newton uses diagonal regularization at that threshold instead.
* ``newton-cold`` starts at the uniform weights. If Newton does not converge,
  it falls back to a freshly run ``fw+bisect`` iterate from that same uniform
  start and reports the fallback's convergence status.
* Newton failure statuses force ``converged=False``; these statuses are
  ``"empty_face"``, ``"armijo_rejected"``, ``"factorization_failed"``,
  ``"ill_conditioned_hessian"``, ``"linear_solve_failed"``,
  ``"nonfinite_residual"``, ``"nonfinite_step"``, and
  ``"projection_failed"``. A full Newton step may additionally be accepted
  for roundoff when
  ``g_trial + 8 * eps * max(1, abs(g_current), abs(g_trial),
  abs(armijo_target)) >= armijo_target``; this allowance applies only to the
  unbacktracked full step.
* The objective supports additive per-constraint offsets, including general
  packed conic constants.
* For stabilization, the SLSQP and damped-Newton objectives are the
  regularized dual rather than ``dot(mu, f)``; this keeps their objective and
  Jacobian consistent with the conditioned matrix.
* A post-polish Cholesky factorization failure forces ``converged=False``;
  ellcech 82d13e3 preserved the prior convergence flag in this case.

The list above exhausts numerical and solver-behavior deviations from ellcech.
The remaining differences are packaging-only (the intra-package import of
``unpack_conic`` and type annotations) or documentation. Distributed as part
of EllPHi under its MIT license; the ellcech source is MIT by owner decision
(uda-lab/project-ellphi#26, 2026-10-04), aligned with EllPHi.

This numerical engine is internal.  The public interface is :mod:`ellphi.cech`.

Mathematical background
-----------------------
Given vertices ``i = 0, ..., k-1`` with centers ``x_i`` in R^d and SPD
matrices ``A_i`` and offsets ``delta_i``, the exact filtration value is

    alpha = min_x  max_i f_i(x),
    f_i(x) = (x - x_i)^T A_i (x - x_i) + delta_i.

``alpha`` is a *squared* scale. In the pairwise case ``k = 2`` the exact
value equals ``ellphi.tangency(p, q).t ** 2``; the tangency time is
``sqrt(alpha)``. Every solver output is a finite-precision, finite-iteration
approximation of these mathematical quantities (within the tolerances and
subject to ``converged``). An *unadjusted* solve (``regularization == 0``
and ``condition_number_limit is None``) approximates the exact problem
directly; passing ``condition_number_limit`` always applies at least a
machine-epsilon diagonal shift, and any positive ``regularization`` shifts
``A(mu)``, so adjusted solves approximate a stabilised problem instead.

By strong duality, ``alpha = max_{mu in Delta^{k-1}} g(mu)`` with

    A(mu) = sum_i mu_i A_i
    b(mu) = sum_i mu_i A_i x_i
    g(mu) = sum_i mu_i (x_i^T A_i x_i + delta_i)
            - b(mu)^T A(mu)^{-1} b(mu)
          = sum_i mu_i f_i(x*(mu)),

where ``x*(mu) = A(mu)^{-1} b(mu)`` is the circumcenter. The dual gradient is
``dg/dmu_i = f_i(x*(mu))`` and the Hessian is
``d2g/dmu_i dmu_j = -2 (x* - x_i)^T A_i A(mu)^{-1} A_j (x* - x_j)``.
These identities describe the unadjusted problem; with regularization or
conditioning the solver works on a stabilised ``A(mu)`` and its outputs
approximate that stabilised problem.

Weight support versus tight set
-------------------------------
``MinimaxResult.active_set`` is the *weight support*
``{i : weights[i] > weight_tol}``. It is not the tight constraint set
``{i : f_i(circumcenter) == alpha}``. At an exact optimum the support of the
weights lies inside the tight set, but the tight set can be strictly larger:
for three unit balls centered at ``(0, 0)``, ``(2, 0)`` and ``(0, 2)``, all
three constraints are tight at the circumcenter ``(1, 1)`` with
``alpha = 2``, while the unique optimal weights are ``(0, 1/2, 1/2)``.

Value versus derivative reliability
-----------------------------------
``MinimaxResult.converged`` reports whether the chosen method met its own
stopping test for the value. It is not a certificate that ``alpha`` is
differentiable at the input or that the returned weights are accurate enough
for derivatives; see :mod:`ellphi._minimax_grad_python` for the hypotheses under
which the gradient formula applies.

Correspondence with the pairwise solver
---------------------------------------
For ``k = 2`` the pairwise solver finds the root of
``F(mu) = f_i(x*(mu)) - f_j(x*(mu))``, while this module maximises ``g``
over ``Delta^1``; both give the same ``x*`` and ``dg/dmu = -F(mu)``. The
pairwise scalar ``mu`` and the vector weights here are related primitives
with different result contracts.

Packed coefficients store the linear term as ``b = -A x_bar``, so the center
is ``x_bar = -A^{-1} b`` and ``delta = c - x_bar^T A x_bar``;
:func:`solve_minimax_from_coefs` handles both conversions.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal, NamedTuple

import numpy as np
import numpy.typing as npt
from scipy import linalg

from .geometry import unpack_conic

__all__ = [
    "MethodName",
    "MinimaxResult",
    "solve_minimax",
    "solve_minimax_from_coefs",
]

MethodName = Literal[
    "fw+bisect",
    "fw+brentq",
    "fw+bisect+newton",
    "fw+brentq+newton",
    "fw+bisect+damped-newton",
    "scipy-slsqp",
    "newton-cold",
]

_DEFAULT_TOL = 1e-9
_DEFAULT_MAX_ITER = 2000
_DEFAULT_WEIGHT_TOL = 1e-10
_DEFAULT_MAX_COND_STEPS = 8
_N_BISECT = 52  # ~machine precision for double

_NEWTON_TOL = 1e-14
_NEWTON_MAX_ITER = 20
_NEWTON_HESSIAN_COND_LIMIT = 1e12

# Brentq line search constants (analogous to ellphi _DEFAULT_HYBRID_BRACKET_MAXITER)
_DEFAULT_BRENTQ_MAXITER = 28
# Retained unused for fidelity with the ellcech source.
_BRENTQ_FAILSAFE_MAXITER = 64
_BRENTQ_XTOL = 1e-12
_BRENTQ_RTOL = 4.0 * np.finfo(float).eps

# Armijo backtracking constants
_ARMIJO_BETA = 0.5
_ARMIJO_SIGMA = 1e-4
_ARMIJO_MAX_BACKTRACK = 10

_VALID_METHODS: tuple[str, ...] = (
    "fw+bisect",
    "fw+brentq",
    "fw+bisect+newton",
    "fw+brentq+newton",
    "fw+bisect+damped-newton",
    "scipy-slsqp",
    "newton-cold",
)

# Legacy alias
_METHOD_ALIASES: dict[str, str] = {
    "fw+newton": "fw+bisect+newton",
}


class MinimaxResult(NamedTuple):
    """Result of the minimax solver.

    Attributes:
        alpha: Filtration value ``max_i f_i(circumcenter)``, a squared scale.
            A numerical approximation of the exact minimax value within the
            solver tolerances (see ``converged``). For an unadjusted solve
            (``regularization == 0`` and ``condition_number_limit is None``)
            the target is the exact problem; otherwise it is the stabilised
            problem. In the pairwise case the exact value is
            ``ellphi.tangency(p, q).t ** 2``.
        circumcenter: Point ``x* = A(mu)^{-1} b(mu)`` for the returned
            weights, computed from the unadjusted ``A(mu)`` for an unadjusted
            solve and from the stabilised ``A(mu)`` otherwise.
        weights: Returned dual weights ``mu`` in the simplex, shape ``(k,)``.
        active_set: Weight support ``{i : weights[i] > weight_tol}``. This is
            not the tight constraint set ``{i : f_i(x*) == alpha}``, which can
            be strictly larger (see the module docstring).
        converged: Whether the method met its stopping test for the value.
            It is not a derivative certificate.
        n_iter: Number of iterations performed.
        method: Canonical solver method used (e.g. ``"fw+bisect"``).
        metadata: Optional dict with method-specific diagnostics
            (``fw_iters``, ``newton_iters``, ``hessian_cond``, ...).
    """

    alpha: float
    circumcenter: np.ndarray
    weights: np.ndarray
    active_set: list[int]
    converged: bool
    n_iter: int
    method: str = "fw+bisect"
    metadata: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Low-level helpers (shared across methods)
# ---------------------------------------------------------------------------


def _condition_matrix(
    A: np.ndarray,
    *,
    regularization: float,
    condition_number_limit: float | None,
    max_conditioning_steps: int,
) -> np.ndarray:
    """Optionally regularise A to improve conditioning."""
    if regularization < 0.0:
        raise ValueError("regularization must be non-negative")
    if condition_number_limit is not None and condition_number_limit <= 1.0:
        raise ValueError("condition_number_limit must be > 1 when provided")
    if max_conditioning_steps < 0:
        raise ValueError("max_conditioning_steps must be non-negative")

    if regularization == 0.0 and condition_number_limit is None:
        return A

    eye = np.eye(A.shape[0], dtype=A.dtype)
    if condition_number_limit is None:
        return A + regularization * eye

    spectral_scale = float(np.linalg.norm(A, ord=2))
    eps_floor = np.finfo(A.dtype).eps * max(1.0, spectral_scale)
    reg = max(regularization, eps_floor)
    A_reg = A + reg * eye

    cond = np.linalg.cond(A_reg)
    if np.isfinite(cond) and cond <= condition_number_limit:
        return A_reg

    # Strict cap: perform at most max_conditioning_steps escalations.
    for _ in range(max_conditioning_steps):
        reg *= 10.0
        A_reg = A + reg * eye
        cond = np.linalg.cond(A_reg)
        if np.isfinite(cond) and cond <= condition_number_limit:
            return A_reg
    return A_reg


def _conditioning_shift_gradient(
    mu: np.ndarray,
    matrices: np.ndarray,
    A_mu: np.ndarray,
    *,
    regularization: float,
    condition_number_limit: float | None,
    max_conditioning_steps: int,
) -> np.ndarray:
    """Differentiate the active scalar conditioning shift with respect to mu."""
    if condition_number_limit is None:
        return np.zeros(mu.shape, dtype=float)

    spectral_scale = float(np.linalg.norm(A_mu, ord=2))
    if spectral_scale < 1.0:
        return np.zeros(mu.shape, dtype=float)
    eps_floor = np.finfo(A_mu.dtype).eps * spectral_scale
    base_shift_from_floor = regularization < eps_floor
    if not base_shift_from_floor:
        base_shift = regularization
        base_gradient = np.zeros(mu.shape, dtype=float)
    else:
        base_shift = eps_floor
        base_gradient = np.empty(mu.shape, dtype=float)
    A_used = _condition_matrix(
        A_mu,
        regularization=regularization,
        condition_number_limit=condition_number_limit,
        max_conditioning_steps=max_conditioning_steps,
    )
    effective_shift = float(np.trace(A_used - A_mu) / A_mu.shape[0])
    escalation = effective_shift / base_shift
    _, eigenvectors = np.linalg.eigh(A_mu)
    leading_vector = eigenvectors[:, -1]
    if base_shift_from_floor:
        spectral_gradient = np.einsum(
            "i,kij,j->k", leading_vector, matrices, leading_vector
        )
        base_gradient = np.finfo(A_mu.dtype).eps * spectral_gradient
        if spectral_scale == 1.0:
            base_gradient *= 0.5
    return escalation * base_gradient


def _validate_tolerance(
    name: str, value: float, *, strictly_positive: bool = False
) -> None:
    """Validate a finite numerical tolerance or regularization value."""
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


def _validate_non_negative_int(name: str, value: int) -> None:
    """Validate a non-negative integer solver parameter."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer")


def _cholesky_solve(
    A: np.ndarray,
    b: np.ndarray,
    *,
    regularization: float = 0.0,
    condition_number_limit: float | None = None,
    max_conditioning_steps: int = _DEFAULT_MAX_COND_STEPS,
) -> np.ndarray:
    """Solve A x = b with optional conditioning safeguards."""
    A = _condition_matrix(
        A,
        regularization=regularization,
        condition_number_limit=condition_number_limit,
        max_conditioning_steps=max_conditioning_steps,
    )
    try:
        chol = linalg.cho_factor(A, check_finite=False)
        return linalg.cho_solve(chol, b, check_finite=False)
    except linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


def _exact_linear_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve A x = b without stability regularization adjustments."""
    try:
        chol = linalg.cho_factor(A, check_finite=False)
        return linalg.cho_solve(chol, b, check_finite=False)
    except linalg.LinAlgError:
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, b, rcond=None)[0]


def _eval_f(
    mu: np.ndarray,
    matrices: np.ndarray,
    Ax: np.ndarray,
    centers: np.ndarray,
    offsets: np.ndarray | None,
    *,
    regularization: float,
    condition_number_limit: float | None,
    max_conditioning_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute circumcenter x*(mu) and f_i(x*(mu)) for all i.

    Also note: g(mu) = sum_i mu_i f_i(x*(mu)) = np.dot(mu, f).
    """
    A_mu = np.einsum("k,kij->ij", mu, matrices)
    b_mu = np.einsum("k,ki->i", mu, Ax)
    xstar = _cholesky_solve(
        A_mu,
        b_mu,
        regularization=regularization,
        condition_number_limit=condition_number_limit,
        max_conditioning_steps=max_conditioning_steps,
    )
    diff = xstar[np.newaxis, :] - centers
    f = np.einsum("ki,kij,kj->k", diff, matrices, diff)
    if offsets is not None:
        f = f + offsets
    return xstar, f


def _dual_gradient(
    mu: np.ndarray,
    matrices: np.ndarray,
    Ax: np.ndarray,
    centers: np.ndarray,
    offsets: np.ndarray | None,
    xstar: np.ndarray,
    f: np.ndarray,
    *,
    regularization: float,
    condition_number_limit: float | None,
    max_conditioning_steps: int,
) -> np.ndarray:
    """Return the gradient of the dual represented by ``xstar`` and ``f``."""
    if regularization == 0.0 and condition_number_limit is None:
        return f

    A_mu = np.einsum("k,kij->ij", mu, matrices)
    shift_gradient = _conditioning_shift_gradient(
        mu,
        matrices,
        A_mu,
        regularization=regularization,
        condition_number_limit=condition_number_limit,
        max_conditioning_steps=max_conditioning_steps,
    )
    return f + shift_gradient * float(np.dot(xstar, xstar))


def _regularized_dual_value(
    mu: np.ndarray,
    matrices: np.ndarray,
    Ax: np.ndarray,
    centers: np.ndarray,
    offsets: np.ndarray | None,
    xstar: np.ndarray,
) -> float:
    """Evaluate the dual value for the matrix used to obtain ``xstar``."""
    c = np.einsum("ki,kij,kj->k", centers, matrices, centers)
    if offsets is not None:
        c = c + offsets
    b_mu = np.einsum("k,ki->i", mu, Ax)
    return float(np.dot(mu, c) - np.dot(b_mu, xstar))


def _exact_line_search(
    s: int,
    v: int,
    mu: np.ndarray,
    matrices: np.ndarray,
    Ax: np.ndarray,
    centers: np.ndarray,
    offsets: np.ndarray | None,
    gamma_max: float,
    *,
    regularization: float,
    condition_number_limit: float | None,
    max_conditioning_steps: int,
) -> float:
    """Find gamma* in [0, gamma_max] that maximises g along the pairwise direction.

    Bisects on the stabilized dual directional derivative
    ``h(gamma) = grad_s - grad_v``.
    """
    if gamma_max <= 0.0:
        return 0.0

    def h(gamma: float) -> float:
        mu_g = mu.copy()
        mu_g[s] += gamma
        mu_g[v] -= gamma
        x_g, f_g = _eval_f(
            mu_g,
            matrices,
            Ax,
            centers,
            offsets,
            regularization=regularization,
            condition_number_limit=condition_number_limit,
            max_conditioning_steps=max_conditioning_steps,
        )
        gradient_g = _dual_gradient(
            mu_g,
            matrices,
            Ax,
            centers,
            offsets,
            x_g,
            f_g,
            regularization=regularization,
            condition_number_limit=condition_number_limit,
            max_conditioning_steps=max_conditioning_steps,
        )
        return float(gradient_g[s] - gradient_g[v])

    h0 = h(0.0)
    if h0 <= 0.0:
        return 0.0  # already at or past optimum in this direction

    h_max = h(gamma_max)
    if h_max >= 0.0:
        return gamma_max  # optimum is at the boundary

    lo, hi = 0.0, gamma_max
    for _ in range(_N_BISECT):
        mid = 0.5 * (lo + hi)
        if h(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _ensure_finite(value: float, label: str) -> float:
    """Raise RuntimeError if *value* is NaN or Inf (ellphi pattern)."""
    if not np.isfinite(value):
        raise RuntimeError(f"Non-finite {label} value: {value}")
    return value


def _brentq_line_search(
    s: int,
    v: int,
    mu: np.ndarray,
    matrices: np.ndarray,
    Ax: np.ndarray,
    centers: np.ndarray,
    offsets: np.ndarray | None,
    gamma_max: float,
    *,
    regularization: float,
    condition_number_limit: float | None,
    max_conditioning_steps: int,
) -> tuple[float, int]:
    """Brentq-based line search replacing fixed 52-step bisection.

    Returns:
        (gamma, n_fevals): Optimal step and number of function evaluations.
    """
    if gamma_max <= 0.0:
        return 0.0, 0

    n_fevals = [0]

    def h(gamma: float) -> float:
        n_fevals[0] += 1
        mu_g = mu.copy()
        mu_g[s] += gamma
        mu_g[v] -= gamma
        x_g, f_g = _eval_f(
            mu_g,
            matrices,
            Ax,
            centers,
            offsets,
            regularization=regularization,
            condition_number_limit=condition_number_limit,
            max_conditioning_steps=max_conditioning_steps,
        )
        gradient_g = _dual_gradient(
            mu_g,
            matrices,
            Ax,
            centers,
            offsets,
            x_g,
            f_g,
            regularization=regularization,
            condition_number_limit=condition_number_limit,
            max_conditioning_steps=max_conditioning_steps,
        )
        return float(gradient_g[s] - gradient_g[v])

    h0 = h(0.0)
    if h0 <= 0.0:
        return 0.0, n_fevals[0]

    h_max = h(gamma_max)
    if h_max >= 0.0:
        return gamma_max, n_fevals[0]

    try:
        from scipy.optimize import brentq

        gamma = brentq(
            h,
            0.0,
            gamma_max,
            xtol=_BRENTQ_XTOL,
            rtol=_BRENTQ_RTOL,
            maxiter=_DEFAULT_BRENTQ_MAXITER,
        )
        return float(gamma), n_fevals[0]
    except (ValueError, RuntimeError):
        # Fallback to fixed bisection
        lo, hi = 0.0, gamma_max
        for _ in range(_N_BISECT):
            mid = 0.5 * (lo + hi)
            if h(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi), n_fevals[0]


# ---------------------------------------------------------------------------
# Per-method internal runners
# ---------------------------------------------------------------------------


def _run_fw_bisect(
    matrices: np.ndarray,
    Ax: np.ndarray,
    centers: np.ndarray,
    offsets: np.ndarray | None,
    mu: np.ndarray,
    *,
    tol: float,
    max_iter: int,
    weight_tol: float,
    regularization: float,
    condition_number_limit: float | None,
    max_conditioning_steps: int,
) -> tuple[np.ndarray, bool, int]:
    """Pairwise Frank-Wolfe with exact bisection line search.

    Returns:
        (mu, converged, n_iter)
    """
    converged = False
    n_iter = 0

    for n_iter in range(1, max_iter + 1):
        xstar, f = _eval_f(
            mu,
            matrices,
            Ax,
            centers,
            offsets,
            regularization=regularization,
            condition_number_limit=condition_number_limit,
            max_conditioning_steps=max_conditioning_steps,
        )

        gradient = _dual_gradient(
            mu,
            matrices,
            Ax,
            centers,
            offsets,
            xstar,
            f,
            regularization=regularization,
            condition_number_limit=condition_number_limit,
            max_conditioning_steps=max_conditioning_steps,
        )

        s = int(np.argmax(gradient))
        active_mask = mu > weight_tol
        if not np.any(active_mask):
            active_mask = np.ones_like(mu, dtype=bool)
        gradient_active = np.where(active_mask, gradient, np.inf)
        v = int(np.argmin(gradient_active))
        if s == v:
            positive_mask = mu > 0.0
            positive_mask[s] = False
            if np.any(positive_mask):
                v = int(np.argmin(np.where(positive_mask, gradient, np.inf)))

        fw_gap = float(np.max(gradient) - np.dot(mu, gradient))
        if fw_gap < tol:
            converged = True
            break

        gamma_max = float(mu[v])
        gamma = _exact_line_search(
            s,
            v,
            mu,
            matrices,
            Ax,
            centers,
            offsets,
            gamma_max,
            regularization=regularization,
            condition_number_limit=condition_number_limit,
            max_conditioning_steps=max_conditioning_steps,
        )

        mu = mu.copy()
        mu[s] += gamma
        mu[v] -= gamma
        mu = np.clip(mu, 0.0, None)
        mu /= mu.sum()

    return mu, converged, n_iter


def _run_fw_brentq(
    matrices: np.ndarray,
    Ax: np.ndarray,
    centers: np.ndarray,
    offsets: np.ndarray | None,
    mu: np.ndarray,
    *,
    tol: float,
    max_iter: int,
    weight_tol: float,
    regularization: float,
    condition_number_limit: float | None,
    max_conditioning_steps: int,
) -> tuple[np.ndarray, bool, int, dict]:
    """Pairwise Frank-Wolfe with brentq line search.

    Returns:
        (mu, converged, n_iter, metadata)
    """
    converged = False
    n_iter = 0
    total_line_search_evals = 0
    total_fevals = 0

    for n_iter in range(1, max_iter + 1):
        xstar, f = _eval_f(
            mu,
            matrices,
            Ax,
            centers,
            offsets,
            regularization=regularization,
            condition_number_limit=condition_number_limit,
            max_conditioning_steps=max_conditioning_steps,
        )
        total_fevals += 1

        gradient = _dual_gradient(
            mu,
            matrices,
            Ax,
            centers,
            offsets,
            xstar,
            f,
            regularization=regularization,
            condition_number_limit=condition_number_limit,
            max_conditioning_steps=max_conditioning_steps,
        )

        s = int(np.argmax(gradient))
        active_mask = mu > weight_tol
        if not np.any(active_mask):
            active_mask = np.ones_like(mu, dtype=bool)
        gradient_active = np.where(active_mask, gradient, np.inf)
        v = int(np.argmin(gradient_active))
        if s == v:
            positive_mask = mu > 0.0
            positive_mask[s] = False
            if np.any(positive_mask):
                v = int(np.argmin(np.where(positive_mask, gradient, np.inf)))

        fw_gap = float(np.max(gradient) - np.dot(mu, gradient))
        if fw_gap < tol:
            converged = True
            break

        gamma_max = float(mu[v])
        gamma, ls_evals = _brentq_line_search(
            s,
            v,
            mu,
            matrices,
            Ax,
            centers,
            offsets,
            gamma_max,
            regularization=regularization,
            condition_number_limit=condition_number_limit,
            max_conditioning_steps=max_conditioning_steps,
        )
        total_line_search_evals += ls_evals

        mu = mu.copy()
        mu[s] += gamma
        mu[v] -= gamma
        mu = np.clip(mu, 0.0, None)
        mu /= mu.sum()

    metadata = {
        "fw_iters": n_iter,
        "n_fevals": total_fevals + total_line_search_evals,
        "line_search_evals": total_line_search_evals,
    }
    return mu, converged, n_iter, metadata


def _hessian_g_negative(
    active: list[int],
    xstar: np.ndarray,
    matrices: np.ndarray,
    Amu_inv: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    """Negative Hessian of g on the active face (positive semidefinite).

    neg_H[l, l'] = 2 (x* - x_{i_l})^T A_{i_l}  A(mu)^{-1}  A_{i_{l'}} (x* - x_{i_{l'}})

    Equivalently neg_H = 2 * Ad @ Amu_inv @ Ad.T
    where Ad[l] = A_{i_l} (x* - x_{i_l})  (shape m x d).
    """
    diff = xstar[np.newaxis, :] - centers[active]  # (m, d)
    Ad = np.einsum("lij,lj->li", matrices[active], diff)  # (m, d)
    return 2.0 * (Ad @ Amu_inv @ Ad.T)  # (m, m)


def _newton_polish(
    mu: np.ndarray,
    active_set: list[int],
    matrices: np.ndarray,
    Ax: np.ndarray,
    centers: np.ndarray,
    offsets: np.ndarray | None,
    *,
    weight_tol: float = _DEFAULT_WEIGHT_TOL,
    max_iter: int = _NEWTON_MAX_ITER,
    tol: float = _NEWTON_TOL,
    regularization: float = 0.0,
    condition_number_limit: float | None = None,
    max_conditioning_steps: int = _DEFAULT_MAX_COND_STEPS,
) -> tuple[np.ndarray, int, dict]:
    """Newton polishing on the dual restricted to the current active face.

    Solves the KKT conditions r_l = f_{i_l}(x*) - f_{i_{m-1}}(x*) = 0
    using the exact (negative) Hessian of g via the reduced system:

        neg_H_r  delta = r
        neg_H_r[l, l'] = neg_H[l,l'] - neg_H[l, m-1] - neg_H[m-1, l'] + neg_H[m-1, m-1]

    Then  mu_{i_l} += delta[l],  mu_{i_{m-1}} -= sum(delta),
    followed by simplex projection.

    Returns:
        Updated weights, number of Newton steps, and diagnostics.
    """
    m = len(active_set)
    if m == 0:
        return (
            mu,
            0,
            {
                "newton_iters": 0,
                "hessian_cond": 0.0,
                "newton_status": "empty_face",
            },
        )
    if m <= 1:
        return (
            mu,
            0,
            {
                "newton_iters": 0,
                "hessian_cond": 0.0,
                "newton_status": "singleton_face",
            },
        )

    mu = mu.copy()
    n_iter = 0
    max_hessian_cond = 0.0
    newton_status = "max_iter"

    for n_iter in range(1, max_iter + 1):
        m = len(active_set)
        if m == 0:
            newton_status = "empty_face"
            break
        if m <= 1:
            newton_status = "singleton_face"
            break

        A_mu = np.einsum("k,kij->ij", mu, matrices)
        b_mu = np.einsum("k,ki->i", mu, Ax)
        A_used = _condition_matrix(
            A_mu,
            regularization=regularization,
            condition_number_limit=condition_number_limit,
            max_conditioning_steps=max_conditioning_steps,
        )

        try:
            chol = linalg.cho_factor(A_used, check_finite=False)
            xstar = linalg.cho_solve(chol, b_mu, check_finite=False)
            d = A_used.shape[0]
            Amu_inv = linalg.cho_solve(chol, np.eye(d), check_finite=False)
        except linalg.LinAlgError:
            newton_status = "factorization_failed"
            break

        diff = xstar[np.newaxis, :] - centers[active_set]  # (m, d)
        f = np.einsum("ki,kij,kj->k", diff, matrices[active_set], diff)  # (m,)
        if offsets is not None:
            f = f + offsets[active_set]

        # Residual: r_l = f_{i_l} - f_{i_{m-1}}
        r = f[:-1] - f[-1]  # (m-1,)
        if float(np.max(np.abs(r))) < tol:
            newton_status = "converged"
            break

        neg_H = _hessian_g_negative(
            active_set, xstar, matrices, Amu_inv, centers
        )  # (m, m)

        # Reduced system (eliminate last variable via sum=1 constraint)
        neg_H_r = (
            neg_H[:-1, :-1] - neg_H[:-1, -1:] - neg_H[-1:, :-1] + neg_H[-1, -1]
        )  # (m-1, m-1)

        hess_cond = float(np.linalg.cond(neg_H_r))
        max_hessian_cond = max(max_hessian_cond, hess_cond)
        if not np.isfinite(hess_cond) or hess_cond > _NEWTON_HESSIAN_COND_LIMIT:
            newton_status = "ill_conditioned_hessian"
            break

        try:
            delta = np.linalg.solve(neg_H_r, r)
        except np.linalg.LinAlgError:
            newton_status = "linear_solve_failed"
            break

        mu_new = mu.copy()
        for pos, idx in enumerate(active_set[:-1]):
            mu_new[idx] += delta[pos]
        mu_new[active_set[-1]] -= float(delta.sum())

        # Project back to simplex
        mu_new = np.clip(mu_new, 0.0, None)
        s = mu_new.sum()
        if s <= 0.0:
            newton_status = "projection_failed"
            break
        mu_new /= s
        mu = mu_new
        active_set = [i for i in range(len(mu)) if mu[i] > weight_tol]

    metadata = {
        "newton_iters": n_iter,
        "hessian_cond": max_hessian_cond,
        "newton_status": newton_status,
    }
    return mu, n_iter, metadata


def _damped_newton_polish(
    mu: np.ndarray,
    active_set: list[int],
    matrices: np.ndarray,
    Ax: np.ndarray,
    centers: np.ndarray,
    offsets: np.ndarray | None,
    *,
    weight_tol: float = _DEFAULT_WEIGHT_TOL,
    reidentify_face: bool = True,
    max_iter: int = _NEWTON_MAX_ITER,
    tol: float = _NEWTON_TOL,
    regularization: float = 0.0,
    condition_number_limit: float | None = None,
    max_conditioning_steps: int = _DEFAULT_MAX_COND_STEPS,
) -> tuple[np.ndarray, int, dict]:
    """Newton polishing with Armijo backtracking and conditioning safeguards.

    Like ``_newton_polish`` but adds:
    - Condition number check on the reduced Hessian (regularize if > 1e12)
    - Armijo backtracking line search on the Newton step
    - ``_ensure_finite`` guards against NaN propagation

    Returns:
        (mu, n_iter, metadata)
    """
    m = len(active_set)
    if m == 0:
        return (
            mu,
            0,
            {
                "newton_iters": 0,
                "hessian_cond": 0.0,
                "newton_status": "empty_face",
            },
        )
    if m <= 1:
        return (
            mu,
            0,
            {
                "newton_iters": 0,
                "hessian_cond": 0.0,
                "newton_status": "singleton_face",
            },
        )

    mu = mu.copy()
    best_mu = mu.copy()
    best_g = -np.inf
    n_iter = 0
    max_hessian_cond = 0.0
    newton_status = "max_iter"
    rejected_step_diagnostics: dict[str, float] = {}
    roundoff_accepts = 0
    max_armijo_roundoff_shortfall = 0.0

    for n_iter in range(1, max_iter + 1):
        m = len(active_set)
        if m == 0:
            newton_status = "empty_face"
            break
        if m <= 1:
            newton_status = "singleton_face"
            break

        A_mu = np.einsum("k,kij->ij", mu, matrices)
        b_mu = np.einsum("k,ki->i", mu, Ax)
        A_used = _condition_matrix(
            A_mu,
            regularization=regularization,
            condition_number_limit=condition_number_limit,
            max_conditioning_steps=max_conditioning_steps,
        )

        try:
            chol = linalg.cho_factor(A_used, check_finite=False)
            xstar = linalg.cho_solve(chol, b_mu, check_finite=False)
            d = A_used.shape[0]
            Amu_inv = linalg.cho_solve(chol, np.eye(d), check_finite=False)
        except linalg.LinAlgError:
            newton_status = "factorization_failed"
            break

        diff = xstar[np.newaxis, :] - centers[active_set]
        f = np.einsum("ki,kij,kj->k", diff, matrices[active_set], diff)
        if offsets is not None:
            f = f + offsets[active_set]

        # Residual
        r = f[:-1] - f[-1]
        try:
            _ensure_finite(float(np.max(np.abs(r))), "Newton residual")
        except RuntimeError:
            newton_status = "nonfinite_residual"
            break
        if float(np.max(np.abs(r))) < tol:
            newton_status = "converged"
            break

        neg_H = _hessian_g_negative(active_set, xstar, matrices, Amu_inv, centers)

        neg_H_r = neg_H[:-1, :-1] - neg_H[:-1, -1:] - neg_H[-1:, :-1] + neg_H[-1, -1]

        # Conditioning check
        hess_cond = float(np.linalg.cond(neg_H_r))
        max_hessian_cond = max(max_hessian_cond, hess_cond)
        if hess_cond > _NEWTON_HESSIAN_COND_LIMIT:
            # Diagonal regularization
            reg = float(np.trace(neg_H_r)) / neg_H_r.shape[0] * 1e-10
            reg = max(reg, 1e-14)
            neg_H_r = neg_H_r + reg * np.eye(neg_H_r.shape[0])

        try:
            delta = np.linalg.solve(neg_H_r, r)
        except np.linalg.LinAlgError:
            newton_status = "linear_solve_failed"
            break

        if not np.all(np.isfinite(delta)):
            newton_status = "nonfinite_step"
            break

        # Current objective: g(mu) = dot(mu, f_full)
        _, f_full = _eval_f(
            mu,
            matrices,
            Ax,
            centers,
            offsets,
            regularization=regularization,
            condition_number_limit=condition_number_limit,
            max_conditioning_steps=max_conditioning_steps,
        )
        if regularization == 0.0 and condition_number_limit is None:
            g_current = float(np.dot(mu, f_full))
        else:
            g_current = _regularized_dual_value(
                mu, matrices, Ax, centers, offsets, xstar
            )
        if g_current > best_g:
            best_g = g_current
            best_mu = mu.copy()

        # Armijo backtracking
        step = 1.0
        accepted = False
        for backtrack in range(_ARMIJO_MAX_BACKTRACK):
            mu_trial = mu.copy()
            for pos, idx in enumerate(active_set[:-1]):
                mu_trial[idx] += step * delta[pos]
            mu_trial[active_set[-1]] -= step * float(delta.sum())

            # Project to simplex
            mu_trial = np.clip(mu_trial, 0.0, None)
            s = mu_trial.sum()
            if s <= 0.0:
                step *= _ARMIJO_BETA
                continue
            mu_trial /= s

            xstar_trial, f_trial = _eval_f(
                mu_trial,
                matrices,
                Ax,
                centers,
                offsets,
                regularization=regularization,
                condition_number_limit=condition_number_limit,
                max_conditioning_steps=max_conditioning_steps,
            )
            if regularization == 0.0 and condition_number_limit is None:
                g_trial = float(np.dot(mu_trial, f_trial))
            else:
                g_trial = _regularized_dual_value(
                    mu_trial, matrices, Ax, centers, offsets, xstar_trial
                )
            armijo_target = g_current + _ARMIJO_SIGMA * step * float(r @ delta)
            armijo_roundoff = (
                8.0
                * np.finfo(float).eps
                * max(1.0, abs(g_current), abs(g_trial), abs(armijo_target))
            )
            if backtrack == 0:
                rejected_step_diagnostics = {
                    "armijo_g_current": g_current,
                    "armijo_full_step_g": g_trial,
                    "armijo_full_step_target": armijo_target,
                    "armijo_residual": float(np.max(np.abs(r))),
                }

            # Armijo sufficient increase (maximizing g)
            roundoff_accepted = (
                backtrack == 0 and g_trial + armijo_roundoff >= armijo_target
            )
            if g_trial >= armijo_target or roundoff_accepted:
                if roundoff_accepted and g_trial < armijo_target:
                    roundoff_accepts += 1
                    max_armijo_roundoff_shortfall = max(
                        max_armijo_roundoff_shortfall, armijo_target - g_trial
                    )
                mu = mu_trial
                if g_trial > best_g:
                    best_g = g_trial
                    best_mu = mu.copy()
                accepted = True
                break
            step *= _ARMIJO_BETA

        if not accepted:
            newton_status = "armijo_rejected"
            mu = best_mu
            break

        if reidentify_face:
            active_set = [i for i in range(len(mu)) if mu[i] > weight_tol]

    metadata = {
        "newton_iters": n_iter,
        "hessian_cond": max_hessian_cond,
        "newton_status": newton_status,
        "armijo_roundoff_accepts": roundoff_accepts,
        "armijo_roundoff_shortfall": max_armijo_roundoff_shortfall,
    }
    if newton_status == "armijo_rejected":
        metadata.update(rejected_step_diagnostics)
    return mu, n_iter, metadata


def _simplex_residual(mu: npt.NDArray[np.float64]) -> float:
    """Equality constraint ``sum(mu) - 1`` for SLSQP."""
    return float(mu.sum() - 1.0)


def _slsqp_objective_and_gradient(
    mu: npt.NDArray[np.float64],
    matrices: np.ndarray,
    Ax: np.ndarray,
    centers: np.ndarray,
    offsets: np.ndarray | None,
    *,
    regularization: float,
    condition_number_limit: float | None,
    max_conditioning_steps: int,
) -> tuple[float, np.ndarray]:
    """Return SLSQP's negative dual objective and its negative gradient."""
    xstar, f = _eval_f(
        mu,
        matrices,
        Ax,
        centers,
        offsets,
        regularization=regularization,
        condition_number_limit=condition_number_limit,
        max_conditioning_steps=max_conditioning_steps,
    )
    gradient = _dual_gradient(
        mu,
        matrices,
        Ax,
        centers,
        offsets,
        xstar,
        f,
        regularization=regularization,
        condition_number_limit=condition_number_limit,
        max_conditioning_steps=max_conditioning_steps,
    )
    if regularization == 0.0 and condition_number_limit is None:
        # Keep the unregularized objective path bitwise identical.
        return -float(np.dot(mu, f)), -gradient

    # The stabilized dual is g_r(mu) = sum_i mu_i c_i -
    # b(mu)^T A_used(mu)^(-1) b(mu). Its gradient includes the derivative
    # of any conditioning shift that depends on A(mu).
    g_regularized = _regularized_dual_value(mu, matrices, Ax, centers, offsets, xstar)
    return -g_regularized, -gradient


def _run_scipy_slsqp(
    matrices: np.ndarray,
    Ax: np.ndarray,
    centers: np.ndarray,
    offsets: np.ndarray | None,
    k: int,
    *,
    regularization: float,
    condition_number_limit: float | None,
    max_conditioning_steps: int,
) -> tuple[np.ndarray, bool, int]:
    """Maximize g(mu) over the simplex using scipy SLSQP.

    Returns:
        (mu, converged, n_iter)
    """
    from scipy.optimize import minimize

    n_eval = [0]

    def neg_g_and_grad(mu: npt.NDArray[np.float64]) -> tuple[float, np.ndarray]:
        n_eval[0] += 1
        return _slsqp_objective_and_gradient(
            mu,
            matrices,
            Ax,
            centers,
            offsets,
            regularization=regularization,
            condition_number_limit=condition_number_limit,
            max_conditioning_steps=max_conditioning_steps,
        )

    x0 = np.full(k, 1.0 / k)
    bounds = [(0.0, None)] * k

    res = minimize(
        neg_g_and_grad,
        x0,
        jac=True,
        method="SLSQP",
        bounds=bounds,
        constraints={"type": "eq", "fun": _simplex_residual},
        options={"ftol": 1e-14, "maxiter": 500},
    )

    mu = np.clip(res.x, 0.0, None)
    s = mu.sum()
    if s > 0.0:
        mu /= s

    return mu, bool(res.success), n_eval[0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _check_newton_convergence(
    mu: np.ndarray,
    active_set: list[int],
    matrices: np.ndarray,
    Ax: np.ndarray,
    centers: np.ndarray,
    offsets: np.ndarray | None,
    converged: bool,
    newton_tol: float,
    newton_status: str,
    *,
    regularization: float = 0.0,
    condition_number_limit: float | None = None,
    max_conditioning_steps: int = _DEFAULT_MAX_COND_STEPS,
) -> bool:
    """Recheck convergence after Newton polishing."""
    failure_statuses = {
        "empty_face",
        "armijo_rejected",
        "factorization_failed",
        "ill_conditioned_hessian",
        "linear_solve_failed",
        "nonfinite_residual",
        "nonfinite_step",
        "projection_failed",
    }
    if not converged or newton_status in failure_statuses:
        return False
    try:
        A_mu = np.einsum("k,kij->ij", mu, matrices)
        b_mu = np.einsum("k,ki->i", mu, Ax)
        A_used = _condition_matrix(
            A_mu,
            regularization=regularization,
            condition_number_limit=condition_number_limit,
            max_conditioning_steps=max_conditioning_steps,
        )
        chol = linalg.cho_factor(A_used, check_finite=False)
        xstar_check = linalg.cho_solve(chol, b_mu, check_finite=False)
        d_check = xstar_check[np.newaxis, :] - centers
        f_check = np.einsum("ki,kij,kj->k", d_check, matrices, d_check)
        if offsets is not None:
            f_check = f_check + offsets
        if len(active_set) <= 1:
            return True
        f_active = f_check[active_set]
        r_check = f_active[:-1] - f_active[-1]
        return float(np.max(np.abs(r_check))) < newton_tol * 1e4
    except linalg.LinAlgError:
        return False


def solve_minimax(
    matrices: np.ndarray,
    centers: np.ndarray,
    *,
    offsets: np.ndarray | None = None,
    method: MethodName | str = "fw+bisect",
    tol: float = _DEFAULT_TOL,
    max_iter: int = _DEFAULT_MAX_ITER,
    weight_tol: float = _DEFAULT_WEIGHT_TOL,
    regularization: float = 0.0,
    condition_number_limit: float | None = None,
    max_conditioning_steps: int = _DEFAULT_MAX_COND_STEPS,
    newton_tol: float = _NEWTON_TOL,
    newton_max_iter: int = _NEWTON_MAX_ITER,
) -> MinimaxResult:
    """Compute a numerical approximation of the filtration value.

    The mathematical target is ``alpha = max_{mu in Delta} g(mu)``. The
    returned value and circumcenter are finite-iteration approximations
    within the solver tolerances; check ``converged``. For an unadjusted
    solve (``regularization == 0`` and ``condition_number_limit is None``)
    the target is the exact problem. Otherwise ``A(mu)`` is stabilised
    (``condition_number_limit`` alone already applies at least a
    machine-epsilon diagonal shift) and the outputs approximate the
    stabilised problem.

    Seven solver back-ends are available via ``method``:

    * ``"fw+bisect"`` (default): Pairwise FW with 52-step bisection line search.
    * ``"fw+brentq"``: Pairwise FW with adaptive brentq line search.
    * ``"fw+bisect+newton"``: FW(bisect) warm-start + Newton polishing.
    * ``"fw+brentq+newton"``: FW(brentq) warm-start + damped Newton polishing.
    * ``"fw+bisect+damped-newton"``: FW(bisect) + Armijo-damped Newton.
    * ``"scipy-slsqp"``: Direct SLSQP solve via scipy.
    * ``"newton-cold"``: Newton from uniform mu=1/k (stability baseline).

    The legacy name ``"fw+newton"`` is accepted as an alias for
    ``"fw+bisect+newton"`` with a deprecation warning.

    Args:
        matrices: SPD matrices ``A_i``, shape ``(k, d, d)``, or ``(d, d)`` for
            a single vertex.
        centers: Centers ``x_i``, shape ``(k, d)``, or ``(d,)`` for a single
            vertex.
        offsets: Optional additive constants ``delta_i``, shape ``(k,)``.
            ``None`` is the original zero-offset problem.
        method: One of the seven canonical method names above.
        tol: Frank-Wolfe gap tolerance for the FW-based methods.
        max_iter: Maximum number of Frank-Wolfe iterations.
        weight_tol: Threshold defining the weight support ``active_set``. It
            also selects the face used by Newton polishing.
        regularization: Non-negative diagonal shift applied to ``A(mu)`` in
            Frank-Wolfe evaluations and line searches, SLSQP evaluations,
            Newton polishing, and the final evaluation.
        condition_number_limit: Optional target for ``cond(A(mu))``. If the
            target is exceeded, the diagonal shift is escalated by factors of
            10 for at most ``max_conditioning_steps`` attempts; the target is
            not guaranteed after the cap.
        max_conditioning_steps: Maximum number of attempted conditioning
            escalations.
        newton_tol: Residual tolerance for Newton polishing.
        newton_max_iter: Maximum number of Newton steps.

    Returns:
        MinimaxResult with ``.method`` recording the canonical solver name
        and ``.metadata`` containing method-specific diagnostics. ``alpha``
        and ``circumcenter`` approximate the exact minimax value and
        circumcenter for an unadjusted solve (``regularization == 0`` and
        ``condition_number_limit is None``), and the stabilised problem
        otherwise; in the pairwise case the exact value is
        ``ellphi.tangency(p, q).t ** 2``. For ``k = 1`` the result uses the
        same conditioned evaluation as the non-singleton paths and keeps
        the trivial converged, zero-iteration semantics.

    Raises:
        ValueError: For an unknown method, an empty simplex or invalid
            conditioning parameters.
    """
    # Handle legacy alias
    if method in _METHOD_ALIASES:
        canonical = _METHOD_ALIASES[method]
        warnings.warn(
            f"Method {method!r} is deprecated, use {canonical!r} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        method = canonical

    if method not in _VALID_METHODS:
        raise ValueError(f"Unknown method {method!r}. Valid methods: {_VALID_METHODS}")

    matrices = np.asarray(matrices, dtype=float)
    centers = np.asarray(centers, dtype=float)
    # Accept single-simplex input (2D matrix, 1D center)
    if matrices.ndim == 2:
        matrices = matrices[np.newaxis]
        centers = centers[np.newaxis]

    k, d = centers.shape

    _validate_tolerance("tol", tol, strictly_positive=True)
    _validate_tolerance("newton_tol", newton_tol)
    _validate_tolerance("regularization", regularization)
    _validate_tolerance("weight_tol", weight_tol)
    _validate_positive_int("max_iter", max_iter)
    _validate_positive_int("newton_max_iter", newton_max_iter)
    _validate_non_negative_int("max_conditioning_steps", max_conditioning_steps)
    if condition_number_limit is not None and (
        not np.isfinite(condition_number_limit) or condition_number_limit <= 1.0
    ):
        raise ValueError("condition_number_limit must be finite and > 1 when provided")

    if k == 0:
        raise ValueError("simplex must contain at least one vertex (k=0 given)")
    if weight_tol >= 1.0 / k:
        raise ValueError("weight_tol must satisfy 0 <= weight_tol < 1/k")
    if offsets is not None:
        offsets = np.asarray(offsets, dtype=float)
        if offsets.shape != (k,):
            raise ValueError(f"offsets shape {offsets.shape} inconsistent with k={k}")
        if not np.all(np.isfinite(offsets)):
            raise ValueError("offsets must be finite")

    # Precompute A_i x_bar_i (shape: k x d)
    Ax = np.einsum("kij,kj->ki", matrices, centers)

    # --- Trivial case: single point ---
    if k == 1:
        xstar, f = _eval_f(
            np.array([1.0]),
            matrices,
            Ax,
            centers,
            offsets,
            regularization=regularization,
            condition_number_limit=condition_number_limit,
            max_conditioning_steps=max_conditioning_steps,
        )
        return MinimaxResult(
            alpha=float(f[0]),
            circumcenter=xstar,
            weights=np.array([1.0]),
            active_set=[0],
            converged=True,
            n_iter=0,
            method=method,
        )

    # Uniform initialisation for FW-based methods
    mu_init = np.full(k, 1.0 / k)

    _fw_kwargs: dict[str, Any] = dict(
        tol=tol,
        max_iter=max_iter,
        weight_tol=weight_tol,
        regularization=regularization,
        condition_number_limit=condition_number_limit,
        max_conditioning_steps=max_conditioning_steps,
    )

    metadata: dict[str, Any] = {}

    # --- Dispatch ---
    if method == "fw+bisect":
        mu, converged, n_iter = _run_fw_bisect(
            matrices, Ax, centers, offsets, mu_init, **_fw_kwargs
        )
        metadata["fw_iters"] = n_iter

    elif method == "fw+brentq":
        mu, converged, n_iter, meta_fw = _run_fw_brentq(
            matrices, Ax, centers, offsets, mu_init, **_fw_kwargs
        )
        metadata.update(meta_fw)

    elif method == "fw+bisect+newton":
        mu, converged, n_iter_fw = _run_fw_bisect(
            matrices, Ax, centers, offsets, mu_init, **_fw_kwargs
        )
        active_set_fw = [i for i in range(k) if mu[i] > weight_tol]
        mu, n_iter_newton, meta_newton = _newton_polish(
            mu,
            active_set_fw,
            matrices,
            Ax,
            centers,
            offsets,
            weight_tol=weight_tol,
            max_iter=newton_max_iter,
            tol=newton_tol,
            regularization=regularization,
            condition_number_limit=condition_number_limit,
            max_conditioning_steps=max_conditioning_steps,
        )
        n_iter = n_iter_fw + n_iter_newton
        active_set_newton = [i for i in range(k) if mu[i] > weight_tol]
        converged = _check_newton_convergence(
            mu,
            active_set_newton,
            matrices,
            Ax,
            centers,
            offsets,
            converged,
            newton_tol,
            meta_newton["newton_status"],
            regularization=regularization,
            condition_number_limit=condition_number_limit,
            max_conditioning_steps=max_conditioning_steps,
        )
        metadata["fw_iters"] = n_iter_fw
        metadata.update(meta_newton)

    elif method == "fw+brentq+newton":
        mu, converged, n_iter_fw, meta_fw = _run_fw_brentq(
            matrices, Ax, centers, offsets, mu_init, **_fw_kwargs
        )
        active_set_fw = [i for i in range(k) if mu[i] > weight_tol]
        mu, n_iter_newton, meta_newton = _damped_newton_polish(
            mu,
            active_set_fw,
            matrices,
            Ax,
            centers,
            offsets,
            weight_tol=weight_tol,
            max_iter=newton_max_iter,
            tol=newton_tol,
            regularization=regularization,
            condition_number_limit=condition_number_limit,
            max_conditioning_steps=max_conditioning_steps,
        )
        n_iter = n_iter_fw + n_iter_newton
        active_set_newton = [i for i in range(k) if mu[i] > weight_tol]
        converged = _check_newton_convergence(
            mu,
            active_set_newton,
            matrices,
            Ax,
            centers,
            offsets,
            converged,
            newton_tol,
            meta_newton["newton_status"],
            regularization=regularization,
            condition_number_limit=condition_number_limit,
            max_conditioning_steps=max_conditioning_steps,
        )
        metadata.update(meta_fw)
        metadata.update(meta_newton)

    elif method == "fw+bisect+damped-newton":
        mu, converged, n_iter_fw = _run_fw_bisect(
            matrices, Ax, centers, offsets, mu_init, **_fw_kwargs
        )
        active_set_fw = [i for i in range(k) if mu[i] > weight_tol]
        mu, n_iter_newton, meta_newton = _damped_newton_polish(
            mu,
            active_set_fw,
            matrices,
            Ax,
            centers,
            offsets,
            weight_tol=weight_tol,
            max_iter=newton_max_iter,
            tol=newton_tol,
            regularization=regularization,
            condition_number_limit=condition_number_limit,
            max_conditioning_steps=max_conditioning_steps,
        )
        n_iter = n_iter_fw + n_iter_newton
        active_set_newton = [i for i in range(k) if mu[i] > weight_tol]
        converged = _check_newton_convergence(
            mu,
            active_set_newton,
            matrices,
            Ax,
            centers,
            offsets,
            converged,
            newton_tol,
            meta_newton["newton_status"],
            regularization=regularization,
            condition_number_limit=condition_number_limit,
            max_conditioning_steps=max_conditioning_steps,
        )
        metadata["fw_iters"] = n_iter_fw
        metadata.update(meta_newton)

    elif method == "newton-cold":
        # Newton from uniform start (no FW warm-up)
        active_set_cold = list(range(k))
        mu, n_iter_newton, meta_newton = _damped_newton_polish(
            mu_init.copy(),
            active_set_cold,
            matrices,
            Ax,
            centers,
            offsets,
            weight_tol=weight_tol,
            reidentify_face=False,
            max_iter=newton_max_iter,
            tol=newton_tol,
            regularization=regularization,
            condition_number_limit=condition_number_limit,
            max_conditioning_steps=max_conditioning_steps,
        )
        n_iter = n_iter_newton
        active_set_newton = [i for i in range(k) if mu[i] > weight_tol]
        converged = _check_newton_convergence(
            mu,
            active_set_newton,
            matrices,
            Ax,
            centers,
            offsets,
            True,
            newton_tol,
            meta_newton["newton_status"],
            regularization=regularization,
            condition_number_limit=condition_number_limit,
            max_conditioning_steps=max_conditioning_steps,
        )
        if not converged:
            mu, fallback_converged, n_iter_fw = _run_fw_bisect(
                matrices, Ax, centers, offsets, mu_init, **_fw_kwargs
            )
            converged = fallback_converged
            n_iter += n_iter_fw
            meta_newton["fallback_method"] = "fw+bisect"
            meta_newton["fallback_fw_iters"] = n_iter_fw
            meta_newton["fallback_converged"] = fallback_converged
            meta_newton["newton_status"] = "fw_fallback"
        metadata.update(meta_newton)

    elif method == "scipy-slsqp":
        mu, converged, n_iter = _run_scipy_slsqp(
            matrices,
            Ax,
            centers,
            offsets,
            k,
            regularization=regularization,
            condition_number_limit=condition_number_limit,
            max_conditioning_steps=max_conditioning_steps,
        )
        metadata["n_fevals"] = n_iter

    # --- Final evaluation ---
    xstar, f = _eval_f(
        mu,
        matrices,
        Ax,
        centers,
        offsets,
        regularization=regularization,
        condition_number_limit=condition_number_limit,
        max_conditioning_steps=max_conditioning_steps,
    )
    alpha = float(np.max(f))
    if not (np.isfinite(alpha) and np.all(np.isfinite(xstar))):
        converged = False
    active_set = [i for i in range(k) if mu[i] > weight_tol]

    return MinimaxResult(
        alpha=alpha,
        circumcenter=xstar,
        weights=mu,
        active_set=active_set,
        converged=converged,
        n_iter=n_iter,
        method=method,
        metadata=metadata or None,
    )


def solve_minimax_from_coefs(
    coefs: np.ndarray,
    **kwargs: Any,
) -> MinimaxResult:
    """Convenience wrapper: accept ellphi packed conic coefficient vectors.

    Converts ellphi's packed representation (A, b_ellphi, c) to
    (A_i, x_bar_i, delta_i) and calls solve_minimax.

    Sign convention:
        ellphi stores ``b_ellphi = -A x_bar`` (linear term of
        ``x^T A x + 2 b^T x + c``), so ``x_bar = -A^{-1} b_ellphi``.
        Completing the square gives
        ``delta = c - x_bar^T A x_bar``.

    Value correspondence:
        In the pairwise case ``result.alpha`` approximates
        ``ellphi.tangency(coefs[0], coefs[1]).t ** 2`` (exactly equal in the
        mathematical problem; numerically within solver tolerances for an
        unadjusted solve, i.e. ``regularization == 0`` and
        ``condition_number_limit is None``).

    Args:
        coefs: Packed conic coefficient array, shape (k, m) or (m,) for k=1.
        **kwargs: Forwarded to solve_minimax (method, tol, max_iter, weight_tol,
            regularization, condition_number_limit, max_conditioning_steps,
            newton_tol, newton_max_iter).

    Returns:
        MinimaxResult.
    """
    coefs = np.asarray(coefs, dtype=float)
    if coefs.ndim == 1:
        coefs = coefs[np.newaxis]

    A_arr, b_arr, c_arr = unpack_conic(coefs)  # (k,d,d), (k,d), (k,)

    # x_bar_i = -A_i^{-1} b_i  (ellphi sign convention: b = -A x_bar)
    k = A_arr.shape[0]
    centers = np.empty_like(b_arr)
    for i in range(k):
        centers[i] = _exact_linear_solve(A_arr[i], -b_arr[i])

    centered_constants = np.einsum("ki,kij,kj->k", centers, A_arr, centers)
    offsets = c_arr - centered_constants
    return solve_minimax(A_arr, centers, offsets=offsets, **kwargs)
