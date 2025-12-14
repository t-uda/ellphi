from typing import Iterable, Tuple
from ellphi import FloatArray
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
    ellcloud: Iterable[FloatArray],
    *,
    parallel: bool = True,
    n_jobs: int | None = -1,
) -> FloatArray: ...

BackendLiteral = Tuple[str, ...]

def has_cpp_backend() -> bool: ...
def tangency(
    pcoef: FloatArray,
    qcoef: FloatArray,
    *,
    method: MethodName | str = "brentq+newton",
    bracket: Tuple[float, float] = (0.0, 1.0),
    x0: float | None = None,
    backend: str = "auto",
    hybrid_bracket_maxiter: int | None = ...,
    hybrid_newton_maxiter: int | None = ...,
    failsafe: bool = True,
) -> TangencyResult: ...
def pdist_tangency(
    ellcloud: Iterable[FloatArray],
    *,
    parallel: bool = True,
    n_jobs: int | None = -1,
    backend: str = "auto",
) -> FloatArray: ...
