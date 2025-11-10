from typing import Optional, Tuple
import numpy as np
from ellphi.solver import MethodName

__all__ = ["solve_mu_gradients", "solve_mu_numerical_diff"]

def solve_mu_gradients(
    p: np.ndarray,
    q: np.ndarray,
    *,
    mu: Optional[float] = None,
    method: MethodName = "brentq+newton",
    bracket: Tuple[float, float] = (0.0, 1.0),
    x0: Optional[float] = None,
) -> Tuple[float, np.ndarray, np.ndarray]: ...
def solve_mu_numerical_diff(
    p: np.ndarray, q: np.ndarray, h: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]: ...
