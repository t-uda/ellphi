"""Differentiable tangency distances for gradient-based optimisation.

This module exposes ``tangency_grad`` (single-pair gradient) and
``pdist_tangency_grad`` (batch pairwise, with VJP pullback).
"""

from __future__ import annotations

import dataclasses
from itertools import combinations
from typing import Callable

import numpy as np

from .differentiable_solver import solve_mu_gradients
from .solver import tangency

__all__ = ["TangencyGrad", "tangency_grad", "pdist_tangency_grad"]


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
        A :class:`TangencyGrad` with ``t``, ``dt_dp``, and ``dt_dq``.

    Raises:
        ZeroDivisionError: When the pencil derivative ``∂F/∂μ`` vanishes at
            the solution — which occurs for degenerate configurations such as
            identical or concentric nested ellipsoids.  These cases make both
            the implicit-function step in :func:`solve_mu_gradients` and the
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


def pdist_tangency_grad(
    coefs: np.ndarray,
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """Pairwise tangency distances and a VJP (pullback) for all pairs.

    Computes the same condensed distance array as :func:`~ellphi.pdist_tangency`
    and additionally returns a VJP function that maps upstream gradient vectors
    back to per-ellipsoid coefficient gradients.

    Args:
        coefs: Array of shape ``(N, m)`` containing ``N`` ellipsoid coefficient
            vectors.

    Returns:
        A tuple ``(dists, vjp)`` where:

        - ``dists`` is a 1-D array of shape ``(N*(N-1)//2,)`` with pairwise
          tangency distances in the same order as ``scipy.spatial.distance.pdist``.
        - ``vjp`` is a callable ``(grad_dists,) -> grad_coefs`` that accumulates
          upstream gradients into an array of shape ``(N, m)``.
    """
    coefs = np.asarray(coefs, dtype=float)
    if coefs.ndim == 3 and coefs.shape[1] == 1:
        coefs = coefs[:, 0, :]
    if coefs.ndim != 2:
        raise ValueError("Expected coefficient array with shape (N, m)")
    N = len(coefs)
    pairs = list(combinations(range(N), 2))
    dists = np.empty(len(pairs))
    store: list[tuple[int, int, np.ndarray, np.ndarray]] = []

    for k, (i, j) in enumerate(pairs):
        g = tangency_grad(coefs[i], coefs[j])
        dists[k] = g.t
        store.append((i, j, g.dt_dp, g.dt_dq))

    def vjp(grad_dists: np.ndarray) -> np.ndarray:
        g_coefs = np.zeros_like(coefs)
        for k, (i, j, dt_dp, dt_dq) in enumerate(store):
            g_coefs[i] += grad_dists[k] * dt_dp
            g_coefs[j] += grad_dists[k] * dt_dq
        return g_coefs

    return dists, vjp
