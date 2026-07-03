"""Differentiable tangency distances and geometry helpers.

Gradient-based optimisation support for ellphi.

This module exposes ``tangency_grad`` (single-pair gradient),
``pdist_tangency_grad`` (batch pairwise, with VJP pullback), and
``coef_from_cov_grad`` (differentiable coefficient computation).
"""

from __future__ import annotations

import dataclasses
from itertools import combinations
from typing import Callable

import numpy as np

from . import _tangency_cpp as _cpp
from .differentiable_solver import solve_mu_gradients
from .solver import tangency

__all__ = ["TangencyGrad", "tangency_grad", "pdist_tangency_grad", "coef_from_cov_grad"]


@dataclasses.dataclass(slots=True, frozen=True)
class TangencyGrad:
    """Tangency distance and its gradients with respect to both ellipsoids.

    Attributes:
        t:      Tangency distance (same as ``TangencyResult.t``).
        dt_dp:  Gradient of ``t`` w.r.t. ``p`` (shape ``(m,)``).
        dt_dq:  Gradient of ``t`` w.r.t. ``q`` (shape ``(m,)``).
    """

    t: float
    dt_dp: np.ndarray
    dt_dq: np.ndarray


def tangency_grad(p: np.ndarray, q: np.ndarray, **solver_kwargs) -> TangencyGrad:
    """Return the tangency distance and its gradients w.r.t. ``p`` and ``q``.

    Uses the envelope theorem: the chain through the optimal center vanishes,
    so only the explicit dependence of the pencil coefficients on ``p``/``q``
    contributes.

    Args:
        p: Coefficient vector of the first ellipsoid.
        q: Coefficient vector of the second ellipsoid.
        **solver_kwargs: Forwarded to ``tangency()`` (e.g. ``method``,
            ``backend``).

    Returns:
        A [`TangencyGrad`][ellphi.grad.TangencyGrad] with ``t``,
        ``dt_dp``, and ``dt_dq``.

    Raises:
        ZeroDivisionError: When the pencil derivative ``∂F/∂μ`` vanishes at
            the solution — which occurs for degenerate configurations such as
            identical or concentric nested ellipsoids.  These cases make both
            the implicit-function step in `solve_mu_gradients` and the
            ``1/(2t)`` factor in the gradient formula ill-defined.
            Note: ``tangency()`` itself returns a small non-zero ``t`` for
            such inputs (never exactly ``0.0``), so this error surfaces from
            ``solve_mu_gradients`` rather than from the ``1/(2t)`` term.
    """
    p = np.asarray(p, dtype=float).reshape(-1)
    q = np.asarray(q, dtype=float).reshape(-1)

    res = tangency(p, q, **solver_kwargs)
    mu, center, t = res.mu, res.point, res.t

    _, d_mu_dp, d_mu_dq = solve_mu_gradients(p, q, mu=mu)

    # Monomial basis evaluated at center: [x_i*x_j (i<=j), 2*x_k, 1]
    tri_i, tri_j = np.triu_indices(center.shape[0])
    quad_entries = np.where(
        tri_i == tri_j,
        center[tri_i] ** 2,
        2.0 * center[tri_i] * center[tri_j],
    )
    base = np.concatenate([quad_entries, 2.0 * center, [1.0]])

    scalar = float(np.dot(base, q - p))
    inv2t = 0.5 / t

    return TangencyGrad(
        t=t,
        dt_dp=inv2t * ((1.0 - mu) * base + scalar * d_mu_dp),
        dt_dq=inv2t * (mu * base + scalar * d_mu_dq),
    )


def _pdist_tangency_grad_python(
    coefs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure-Python reference: one ``tangency_grad`` call per pair.

    Returns the condensed distances plus the per-pair gradient blocks
    ``dt_dp`` / ``dt_dq`` of shape ``(n_pairs, m)``.
    """
    N, m = coefs.shape
    n_pairs = N * (N - 1) // 2
    dists = np.empty(n_pairs)
    dt_dp = np.empty((n_pairs, m))
    dt_dq = np.empty((n_pairs, m))

    for k, (i, j) in enumerate(combinations(range(N), 2)):
        g = tangency_grad(coefs[i], coefs[j])
        dists[k] = g.t
        dt_dp[k] = g.dt_dp
        dt_dq[k] = g.dt_dq

    return dists, dt_dp, dt_dq


def pdist_tangency_grad(
    coefs: np.ndarray,
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """Pairwise tangency distances and a VJP (pullback) for all pairs.

    Computes the same condensed distance array as
    [`pdist_tangency`][ellphi.pdist_tangency]
    and additionally returns a VJP function that maps upstream gradient vectors
    back to per-ellipsoid coefficient gradients.

    The batched distance/gradient computation runs on the C++ backend when it
    is available (mirroring ``pdist_tangency``) and falls back to the
    pure-Python reference implementation otherwise.  The VJP accumulation
    itself always runs in NumPy.

    Args:
        coefs: Array of shape ``(N, m)`` containing ``N`` ellipsoid coefficient
            vectors.

    Returns:
        A tuple ``(dists, vjp)`` where:

        - ``dists`` is a 1-D array of shape ``(N*(N-1)//2,)`` with pairwise
            tangency distances in the same order as ``scipy.spatial.distance.pdist``.
        - ``vjp`` is a callable ``(grad_dists,) -> grad_coefs`` that accumulates
            upstream gradients into an array of shape ``(N, m)``.

    Examples:
        >>> import numpy as np
        >>> from ellphi import ellipse_cloud
        >>> from ellphi.grad import pdist_tangency_grad
        >>> rng = np.random.default_rng(0)
        >>> cloud = ellipse_cloud(rng.standard_normal((6, 2)), k=3)
        >>> dists, vjp = pdist_tangency_grad(cloud.coef)
        >>> dists.shape        # N*(N-1)//2 = 15
        (15,)
        >>> vjp(np.ones(15)).shape   # (N, m) = (6, 6)
        (6, 6)
    """
    coefs = np.asarray(coefs, dtype=float)
    if coefs.ndim == 3 and coefs.shape[1] == 1:
        coefs = coefs[:, 0, :]
    if coefs.ndim != 2:
        raise ValueError("Expected coefficient array with shape (N, m)")
    N = len(coefs)

    if _cpp.has_pdist_tangency_grad():
        dists, dt_dp, dt_dq = _cpp.pdist_tangency_grad(coefs)
    else:
        dists, dt_dp, dt_dq = _pdist_tangency_grad_python(coefs)

    idx_i, idx_j = np.triu_indices(N, k=1)

    def vjp(grad_dists: np.ndarray) -> np.ndarray:
        gd = np.asarray(grad_dists, dtype=float)
        g_coefs = np.zeros_like(coefs)
        np.add.at(g_coefs, idx_i, gd[:, None] * dt_dp)
        np.add.at(g_coefs, idx_j, gd[:, None] * dt_dq)
        return g_coefs

    return dists, vjp


def coef_from_cov_grad(
    X: np.ndarray,
    cov: np.ndarray,
    /,
    *,
    scale: float = 1.0,
) -> tuple[np.ndarray, Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]]:
    """Converts centers and covariances to packed conic coefficients, with VJP.

    This is the differentiable version of ``coef_from_cov``. It returns the same
    coefficient array plus a VJP (vector-Jacobian product) pullback function.

    Args:
        X: Centers array, shape ``(n, d)`` or ``(d,)``.
        cov: Covariance matrices, shape ``(n, d, d)`` or ``(d, d)``.
        scale: Optional scaling factor for covariance matrices.

    Returns:
        A tuple ``(coefs, vjp)`` where:

        - ``coefs``: shape ``(n, m)``, same as ``coef_from_cov`` output
        - ``vjp``: callable ``(grad_coefs,) -> (grad_X, grad_cov)`` where
          ``grad_X`` has shape ``(n, d)`` and ``grad_cov`` has shape ``(n, d, d)``
    """
    from .geometry import pack_conic

    centers = np.asarray(X, dtype=float)
    cov_arr = np.asarray(cov, dtype=float)

    if centers.ndim == 1:
        centers = centers[np.newaxis, :]
    if cov_arr.ndim == 2:
        cov_arr = cov_arr[np.newaxis, :, :]

    if centers.shape[0] != cov_arr.shape[0]:
        raise ValueError("Mismatch between number of centres and covariance matrices")
    if cov_arr.shape[-1] != cov_arr.shape[-2]:
        raise ValueError("Covariance matrices must be square")

    n, d = centers.shape

    if cov_arr.shape[-1] != d:
        raise ValueError("Centre dimensionality and covariance size must agree")
    s2 = scale**2

    # Forward pass – handle singular covariances the same way coef_from_cov
    # does: return NaN coefficients so callers can detect degeneracy.
    try:
        inv_cov = np.linalg.inv(cov_arr)  # (n, d, d)
    except np.linalg.LinAlgError:
        m = (d + 1) * (d + 2) // 2
        coefs = np.full((n, m), np.nan, dtype=float)

        def vjp_nan(
            grad_coefs: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            return (
                np.full_like(centers, np.nan),
                np.full_like(cov_arr, np.nan),
            )

        return coefs, vjp_nan

    A = inv_cov / s2  # (n, d, d)
    b = -np.einsum("nij,nj->ni", A, centers)  # (n, d)
    c = np.einsum("ni,nij,nj->n", centers, A, centers)  # (n,)

    coefs = pack_conic(A, b, c)

    # Precompute indices for unpacking
    tri_i, tri_j = np.triu_indices(d)
    n_quad = tri_i.size

    def vjp(
        grad_coefs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        gc = np.asarray(grad_coefs, dtype=float)

        # Unpack grad_coefs into grad_A_packed, grad_b, grad_c
        grad_A_packed = gc[:, :n_quad]  # (n, n_quad)
        grad_b = gc[:, n_quad : n_quad + d]  # (n, d)
        grad_c = gc[:, n_quad + d]  # (n,)

        # Reconstruct grad_A from upper-tri packed entries.
        # pack_conic extracts A[tri_i, tri_j] (upper triangle only), so
        # the gradient from packing only touches those entries.
        grad_A = np.zeros((n, d, d), dtype=float)
        grad_A[:, tri_i, tri_j] = grad_A_packed

        # Accumulate contributions from b = -A @ center into grad_A.
        # b_k = -sum_j A[k,j] * center[j], so dL/dA[k,j] += -grad_b[k]*center[j]
        # This contribution is for the FULL matrix (not just upper triangle).
        grad_A += -np.einsum("ni,nj->nij", grad_b, centers)

        # Accumulate contributions from c = center^T A center into grad_A.
        # c = sum_{i,j} center[i] A[i,j] center[j], so
        # dL/dA[i,j] += grad_c * center[i] * center[j]
        grad_A += grad_c[:, None, None] * np.einsum("ni,nj->nij", centers, centers)

        # A is parameterized by symmetric covariances, so project the
        # unconstrained matrix gradient back onto the symmetric subspace before
        # exposing grad_cov or differentiating through inv(cov).
        grad_A = 0.5 * (grad_A + np.swapaxes(grad_A, -2, -1))

        # --- grad w.r.t. centers ---
        # From b = -A @ center: grad_center += -A^T @ grad_b
        grad_centers = -np.einsum("nji,nj->ni", A, grad_b)
        # From c = center^T A center: grad_center += 2 * A @ center * grad_c
        #   = -2 * b * grad_c
        grad_centers += -2.0 * b * grad_c[:, None]

        # --- grad w.r.t. cov (via A = inv(cov)/s^2) ---
        # A = inv(cov) / s^2
        # dA = -inv(cov) @ dCov @ inv(cov) / s^2
        # <grad_A, dA> = tr(grad_A^T (-inv(cov) dCov inv(cov) / s^2))
        #              = -1/s^2 * tr(inv(cov)^T grad_A^T inv(cov)^T dCov)
        # Since inv(cov) and grad_A are symmetric:
        #   grad_cov = -inv(cov) @ grad_A @ inv(cov) / s^2
        #            = -s^2 A @ grad_A @ s^2 A / s^2
        #            = -s^2 * A @ grad_A @ A
        grad_cov = -s2 * np.einsum("nij,njk,nkl->nil", A, grad_A, A)

        return grad_centers, grad_cov

    return coefs, vjp
