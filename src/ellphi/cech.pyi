from typing import NamedTuple

import numpy as np

from ._minimax_python import MethodName

__all__ = [
    "CechResult",
    "CechGrad",
    "cech",
    "cech_grad",
]

class CechResult(NamedTuple):
    t: float
    point: np.ndarray
    mu: np.ndarray
    support: tuple[int, ...]
    active_set: tuple[int, ...]

class CechGrad(NamedTuple):
    t: float
    point: np.ndarray
    mu: np.ndarray
    dt_dcoef: np.ndarray
    support: tuple[int, ...]
    active_set: tuple[int, ...]

def cech(
    coefs: np.ndarray,
    *,
    method: MethodName | str = "fw+brentq",
    tol: float = 1e-9,
    max_iter: int = 2000,
    weight_tol: float = 1e-10,
    active_tol: float = 1e-9,
    newton_tol: float = 1e-14,
    newton_max_iter: int = 20,
) -> CechResult:
    """Compute Čech time using pairwise FW swaps on the dual simplex."""
    ...

def cech_grad(
    coefs: np.ndarray,
    *,
    method: MethodName | str = "fw+brentq",
    tol: float = 1e-9,
    max_iter: int = 2000,
    weight_tol: float = 1e-10,
    active_tol: float = 1e-9,
    newton_tol: float = 1e-14,
    newton_max_iter: int = 20,
) -> CechGrad:
    """Compute Čech time and its gradient using pairwise FW swaps."""
    ...
