from typing import NamedTuple, Tuple
import numpy
from numpy.typing import NDArray
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

def quad_eval(
    coef: NDArray[numpy.floating], center: Tuple[float, ...] | NDArray[numpy.floating]
) -> float: ...
def pencil(
    p: NDArray[numpy.floating], q: NDArray[numpy.floating], mu: float
) -> NDArray[numpy.floating]: ...

class TangencyResult(NamedTuple):
    t: float
    point: NDArray[numpy.floating]
    mu: float

def solve_mu(
    p: NDArray[numpy.floating],
    q: NDArray[numpy.floating],
    *,
    method: MethodName = "brentq+newton",
    bracket: Tuple[float, float] = (0.0, 1.0),
    x0: float | None = None,
) -> float: ...
def tangency(
    pcoef: NDArray[numpy.floating],
    qcoef: NDArray[numpy.floating],
    *,
    method: MethodName = "brentq+newton",
    bracket: Tuple[float, float] = (0.0, 1.0),
    x0: float | None = None,
) -> TangencyResult: ...
def _pdist_tangency_serial(ellcloud: EllipseCloud) -> NDArray[numpy.floating]: ...
def _pdist_tangency_parallel(
    ellcloud: EllipseCloud, n_jobs: int | None = -1
) -> NDArray[numpy.floating]: ...
