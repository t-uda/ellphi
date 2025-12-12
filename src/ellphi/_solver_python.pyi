from typing import NamedTuple, Tuple
from ellphi import FloatArray
from ellphi.ellcloud import EllipseCloud

__all__ = [
    "quad_eval",
    "pencil",
    "TangencyResult",
    "solve_mu",
    "tangency",
    "_pdist_tangency_serial",
    "_pdist_tangency_parallel",
]

MethodName = str  # Simplified for stub

def _gaussian_elimination(matrix: FloatArray, rhs: FloatArray) -> FloatArray: ...
def _center(coef: FloatArray) -> FloatArray: ...
def quad_eval(coef: FloatArray, center: Tuple[float, ...] | FloatArray) -> float: ...
def pencil(p: FloatArray, q: FloatArray, mu: float) -> FloatArray: ...

class TangencyResult(NamedTuple):
    t: float
    point: FloatArray
    mu: float

def solve_mu(
    p: FloatArray,
    q: FloatArray,
    *,
    method: MethodName = "brentq+newton",
    bracket: Tuple[float, float] = (0.0, 1.0),
    x0: float | None = None,
    hybrid_bracket_maxiter: int | None = ...,
    hybrid_newton_maxiter: int | None = ...,
    failsafe: bool = True,
) -> float: ...
def tangency(
    pcoef: FloatArray,
    qcoef: FloatArray,
    *,
    method: MethodName = "brentq+newton",
    bracket: Tuple[float, float] = (0.0, 1.0),
    x0: float | None = None,
    hybrid_bracket_maxiter: int | None = ...,
    hybrid_newton_maxiter: int | None = ...,
    failsafe: bool = True,
) -> TangencyResult: ...
def _pdist_tangency_serial(ellcloud: EllipseCloud) -> FloatArray: ...
def _pdist_tangency_parallel(
    ellcloud: EllipseCloud, n_jobs: int | None = -1
) -> FloatArray: ...
