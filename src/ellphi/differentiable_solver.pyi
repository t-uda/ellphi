from typing import Tuple
from ellphi import FloatArray
from ellphi.solver import MethodName

__all__ = ["solve_mu_gradients", "solve_mu_numerical_diff"]

def solve_mu_gradients(
    p: FloatArray,
    q: FloatArray,
    *,
    mu: float | None = None,
    method: MethodName = "brentq+newton",
    bracket: Tuple[float, float] = (0.0, 1.0),
    x0: float | None = None,
    hybrid_bracket_maxiter: int | None = ...,
    hybrid_newton_maxiter: int | None = ...,
) -> Tuple[float, FloatArray, FloatArray]: ...
def solve_mu_numerical_diff(
    p: FloatArray, q: FloatArray, h: float = 1e-6
) -> Tuple[FloatArray, FloatArray]: ...
