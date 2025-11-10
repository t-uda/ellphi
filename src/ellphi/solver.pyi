from typing import Iterable, Tuple, Optional, Union
import numpy
from numpy.typing import NDArray
from ._solver_python import TangencyResult, MethodName

__all__ = [
    "quad_eval",
    "pencil",
    "TangencyResult",
    "solve_mu",
    "tangency",
    "pdist_tangency",
    "tangency_python",
    "pdist_tangency_python",
    "has_cpp_backend",
    "MethodName",
]

# Re-export from _solver_python
from ._solver_python import (
    quad_eval,
    pencil,
    solve_mu,
    tangency as tangency_python,
)

def pdist_tangency_python(
    ellcloud: Iterable[NDArray[numpy.floating]],
    *,
    parallel: bool = True,
    n_jobs: Optional[int] = -1,
) -> NDArray[numpy.floating]: ...

BackendLiteral = Tuple[str, ...]

def has_cpp_backend() -> bool: ...
def tangency(
    pcoef: NDArray[numpy.floating],
    qcoef: NDArray[numpy.floating],
    *,
    method: Union[MethodName, str] = "brentq+newton",
    bracket: Tuple[float, float] = (0.0, 1.0),
    x0: Optional[float] = None,
    backend: str = "auto",
) -> TangencyResult: ...
def pdist_tangency(
    ellcloud: Iterable[NDArray[numpy.floating]],
    *,
    parallel: bool = True,
    n_jobs: Optional[int] = -1,
    backend: str = "auto",
) -> NDArray[numpy.floating]: ...
