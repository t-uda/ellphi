from typing import Any, Literal, NamedTuple, TypeAlias

import numpy as np

__all__ = [
    "MethodName",
    "MinimaxResult",
    "solve_minimax",
    "solve_minimax_from_coefs",
]

MethodName: TypeAlias = Literal[
    "fw+bisect",
    "fw+brentq",
    "fw+bisect+newton",
    "fw+brentq+newton",
    "fw+bisect+damped-newton",
    "scipy-slsqp",
    "newton-cold",
]

class MinimaxResult(NamedTuple):
    alpha: float
    circumcenter: np.ndarray
    weights: np.ndarray
    active_set: list[int]
    converged: bool
    n_iter: int
    method: str = "fw+bisect"
    metadata: dict[str, Any] | None = None

def solve_minimax(
    matrices: np.ndarray,
    centers: np.ndarray,
    *,
    offsets: np.ndarray | None = None,
    method: MethodName | str = "fw+brentq",
    tol: float = 1e-09,
    max_iter: int = 2000,
    weight_tol: float = 1e-10,
    regularization: float = 0.0,
    condition_number_limit: float | None = None,
    max_conditioning_steps: int = 8,
    newton_tol: float = 1e-14,
    newton_max_iter: int = 20,
) -> MinimaxResult:
    """Solve the dual with plain FW and a robust SLSQP/Newton fallback."""
    ...

def solve_minimax_from_coefs(
    coefs: np.ndarray,
    **kwargs: Any,
) -> MinimaxResult: ...
