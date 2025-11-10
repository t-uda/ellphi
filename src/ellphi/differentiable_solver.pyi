from typing import Optional
import numpy as np
from ellphi.solver import MethodName

__all__ = ["solve_mu_gradients"]

def solve_mu_gradients(
    p: np.ndarray,
    q: np.ndarray,
    *,
    mu: Optional[float] = None,
    method: MethodName = "brentq+newton",
    bracket: tuple[float, float] = (0.0, 1.0),
    x0: Optional[float] = None,
) -> tuple[float, np.ndarray, np.ndarray]: ...
