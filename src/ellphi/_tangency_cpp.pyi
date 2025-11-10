from typing import Tuple
import numpy
from numpy.typing import NDArray
from ._solver_python import TangencyResult

def is_available() -> bool: ...
def tangency(
    pcoef: NDArray[numpy.floating],
    qcoef: NDArray[numpy.floating],
    *,
    method: str,
    bracket: Tuple[float, float],
    x0: float | None
) -> TangencyResult: ...
def pdist_tangency(coef: NDArray[numpy.floating]) -> NDArray[numpy.floating]: ...
