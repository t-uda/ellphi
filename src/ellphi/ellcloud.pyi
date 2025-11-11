from __future__ import annotations
from __future__ import annotations
from typing import Any, Iterator, Sequence
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from ellphi import FloatArray

__all__ = ["ellipse_cloud", "EllipseCloud", "LocalCov"]

@dataclass
class EllipseCloud:
    coef: FloatArray
    mean: FloatArray
    cov: FloatArray
    k: int
    nbd: NDArray[np.int_]
    n: int = field(init=False)
    n_dim: int = field(init=False)
    def __post_init__(self) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[FloatArray]: ...
    def __getitem__(self, idx: Any) -> FloatArray: ...
    def plot(
        self,
        ids: Sequence[int] | None = None,
        ax: plt.Axes | None = None,
        scale: float = 1.0,
        **kwgs: Any,
    ) -> plt.Axes: ...
    def pdist_tangency(
        self, *, parallel: bool = True, n_jobs: int | None = -1, backend: str = "auto"
    ) -> FloatArray: ...
    @classmethod
    def from_point_cloud(
        cls,
        X: FloatArray,
        *,
        method: str = "local_cov",
        rescaling: str = "none",
        **kwgs: Any,
    ) -> EllipseCloud: ...
    @classmethod
    def from_local_cov(cls, X: FloatArray, *, k: int = 5) -> EllipseCloud: ...
    def rescale(self, *, method: str = "median") -> float: ...

def ellipse_cloud(
    X: FloatArray,
    *,
    method: str = "local_cov",
    rescaling: str = "none",
    **kwgs: Any,
) -> EllipseCloud: ...
@dataclass(frozen=True)
class LocalCov:
    k: int = 5
    def __call__(self, X: FloatArray) -> EllipseCloud: ...
