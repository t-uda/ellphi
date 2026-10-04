"""Gradient of the many-body minimax filtration value (provisional).

Provenance
----------
Adapted from uda-lab/ellcech ``src/ellphi_alpha/minimax_grad.py`` at commit
82d13e3e174f4903cdcd6adcee1f469348361c91, by Tomoki Uda (the author of
EllPHi). The only later change to that file on ellcech ``main`` (commit
ebe63a42060ac6420ad42d90b3ffe942969d54ec) is documentation-only. This
docstring is written for EllPHi rather than copied from either revision: it
adopts the ebe63a4 remark that zero weights give zero gradient blocks, but
not its characterization of differentiability. The code is unchanged.
Distributed as part of EllPHi under its MIT license. The rights question for
the ellcech source is tracked in uda-lab/project-ellphi#26.

The module name and the public names exported from ``ellphi`` are
provisional and may change.

Formula
-------
Given a :class:`~ellphi.minimax.MinimaxResult` with weights ``mu`` and
circumcenter ``x*``, the envelope theorem applied to the dual gives

    d alpha / d xbar_i = 2 mu_i A_i (xbar_i - x*)
    d alpha / d A_i    = mu_i outer(xbar_i - x*, xbar_i - x*)

with centers ``xbar_i`` and matrices ``A_i`` treated as independent
parameters. For ``k = 2`` these are the pairwise gradients of ``t ** 2``.
Indices with zero weight get zero blocks; a zero weight alone is not
evidence that ``alpha`` fails to be differentiable.

Hypotheses
----------
The returned arrays are the formula evaluated at the solver output. They
approximate the true gradient of ``alpha`` only under the following
hypotheses:

* the input configuration is non-degenerate;
* at least two indices are active;
* the returned weights are close to the optimal multipliers and are
  supported on the active set;
* the weighted linear solve for ``x*`` is exact. A floating-point solve
  leaves an additional residual.

"Active" in these hypotheses refers to the exact optimization problem, not
to the thresholded ``result.active_set``.

``MinimaxResult.converged`` and the value of ``alpha`` do not certify these
hypotheses. The formula scales each block by ``result.weights[i]`` directly;
it does not use ``result.active_set`` or ``weight_tol``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .minimax import MinimaxResult

__all__ = [
    "GradientResult",
    "compute_gradient",
]


@dataclass
class GradientResult:
    """Gradient of ``alpha`` with respect to centers and metric matrices.

    Attributes:
        d_xbar: Gradients with respect to the centers ``xbar_i``, one array
            of shape ``(d,)`` per vertex.
        d_A: Gradients with respect to the matrices ``A_i``, one symmetric
            array of shape ``(d, d)`` per vertex.
        alpha: The filtration value from the forward pass.
        circumcenter: The circumcenter ``x*`` from the forward pass.
        weights: The weights ``mu`` from the forward pass.
    """

    d_xbar: list[np.ndarray]
    d_A: list[np.ndarray]
    alpha: float
    circumcenter: np.ndarray
    weights: np.ndarray


def compute_gradient(
    result: MinimaxResult,
    centers: np.ndarray | Sequence[np.ndarray],
    matrices: np.ndarray | Sequence[np.ndarray],
) -> GradientResult:
    """Evaluate the envelope-theorem gradient of ``alpha`` at a solver result.

    Computes ``d alpha / d xbar_i = 2 mu_i A_i (xbar_i - x*)`` and
    ``d alpha / d A_i = mu_i outer(xbar_i - x*, xbar_i - x*)``. See the module
    docstring for the hypotheses under which these approximate the true
    gradient.

    Args:
        result: A MinimaxResult from :func:`ellphi.minimax.solve_minimax`
            computed for the same ``centers`` and ``matrices``.
        centers: Centers ``xbar_i``, shape ``(k, d)``.
        matrices: SPD matrices ``A_i``, shape ``(k, d, d)``.

    Returns:
        GradientResult with ``d_xbar`` and ``d_A`` lists of length ``k``.

    Raises:
        ValueError: If the shapes of ``centers``, ``matrices`` and
            ``result.weights`` are inconsistent.
    """
    centers = np.asarray(centers, dtype=float)
    matrices = np.asarray(matrices, dtype=float)

    if centers.ndim == 1:
        centers = centers[np.newaxis]
        matrices = matrices[np.newaxis]

    k, d = centers.shape
    if matrices.shape != (k, d, d):
        raise ValueError(
            f"matrices shape {matrices.shape} inconsistent with "
            f"centers shape {centers.shape}"
        )
    if len(result.weights) != k:
        raise ValueError(
            f"result.weights length {len(result.weights)} does not match k={k}"
        )

    xstar = result.circumcenter  # shape (d,)
    mu = result.weights  # shape (k,)

    d_xbar = []
    d_A = []

    for i in range(k):
        diff = centers[i] - xstar  # xbar_i - x*, shape (d,)
        mu_i = float(mu[i])

        # d alpha / d xbar_i = 2 mu_i A_i (xbar_i - x*)
        grad_xbar_i = 2.0 * mu_i * (matrices[i] @ diff)  # shape (d,)

        # d alpha / d A_i = mu_i outer(xbar_i - x*, xbar_i - x*)
        grad_A_i = mu_i * np.outer(diff, diff)  # shape (d, d)

        d_xbar.append(grad_xbar_i)
        d_A.append(grad_A_i)

    return GradientResult(
        d_xbar=d_xbar,
        d_A=d_A,
        alpha=result.alpha,
        circumcenter=xstar.copy(),
        weights=mu.copy(),
    )
