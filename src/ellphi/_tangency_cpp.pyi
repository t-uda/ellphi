from typing import Tuple
from ellphi import FloatArray
from ._solver_python import TangencyResult

def is_available() -> bool: ...
def tangency(
    pcoef: FloatArray,
    qcoef: FloatArray,
    *,
    method: str,
    bracket: Tuple[float, float],
    x0: float | None,
    hybrid_bracket_maxiter: int,
    hybrid_newton_maxiter: int,
    failsafe: bool,
) -> TangencyResult: ...
def pdist_tangency(coef: FloatArray) -> FloatArray: ...
