from typing import Any, NamedTuple

import numpy as np

__all__ = [
    "SimplexTangencyResult",
    "SimplexTangencyGrad",
    "tangency_simplex",
    "tangency_simplex_grad",
]

class SimplexTangencyResult(NamedTuple):
    t: float
    point: np.ndarray
    mu: np.ndarray
    support: tuple[int, ...]
    active_set: tuple[int, ...]

class SimplexTangencyGrad(NamedTuple):
    t: float
    point: np.ndarray
    mu: np.ndarray
    dt_dcoef: np.ndarray
    support: tuple[int, ...]
    active_set: tuple[int, ...]

def tangency_simplex(
    coefs: np.ndarray,
    *,
    method: str = "fw+bisect",
    tol: float = 1e-9,
    max_iter: int = 2000,
    weight_tol: float = 1e-10,
    active_tol: float = 1e-9,
    norm_tol: float = 1e-9,
    regularization: float = 0.0,
    condition_number_limit: float | None = None,
    max_conditioning_steps: int = 8,
    newton_tol: float = 1e-14,
    newton_max_iter: int = 20,
) -> SimplexTangencyResult: ...
def tangency_simplex_grad(
    coefs: np.ndarray, **solver_kwargs: Any
) -> SimplexTangencyGrad: ...
