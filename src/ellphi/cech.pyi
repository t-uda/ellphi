from typing import Literal, NamedTuple, TypeAlias

import numpy as np

__all__ = [
    "MethodName",
    "CechStage",
    "CechInfo",
    "CechResult",
    "CechGrad",
    "cech",
    "cech_grad",
]

MethodName: TypeAlias = Literal[
    "fw+bisect",
    "fw+brentq",
    "fw+bisect+newton",
    "fw+brentq+newton",
    "fw+bisect+damped-newton",
    "scipy-slsqp",
    "newton-cold",
    "auto",
]

class CechStage(NamedTuple):
    method: str
    converged: bool
    status: str
    gap: float
    n_iter: int

class CechInfo(NamedTuple):
    requested_method: str
    method_used: str
    converged: bool
    gap: float
    n_iter: int
    stages: tuple[CechStage, ...]

class CechResult(NamedTuple):
    t: float
    point: np.ndarray
    mu: np.ndarray
    support: tuple[int, ...]
    active_set: tuple[int, ...]
    info: CechInfo

class CechGrad(NamedTuple):
    t: float
    point: np.ndarray
    mu: np.ndarray
    dt_dcoef: np.ndarray
    support: tuple[int, ...]
    active_set: tuple[int, ...]
    info: CechInfo

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
    """Compute Čech time using pairwise FW swaps on the dual simplex."""
    ...

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
    """Compute Čech time and its gradient using pairwise FW swaps."""
    ...
