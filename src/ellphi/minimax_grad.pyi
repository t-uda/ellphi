from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .minimax import MinimaxResult

__all__ = [
    "GradientResult",
    "compute_gradient",
]

@dataclass
class GradientResult:
    d_xbar: list[np.ndarray]
    d_A: list[np.ndarray]
    alpha: float
    circumcenter: np.ndarray
    weights: np.ndarray

def compute_gradient(
    result: MinimaxResult,
    centers: np.ndarray | Sequence[np.ndarray],
    matrices: np.ndarray | Sequence[np.ndarray],
) -> GradientResult: ...
