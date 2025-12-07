import numpy as np
from .geometry import unpack_single_conic as unpack_single_conic
from dataclasses import dataclass
from typing import Tuple, Any
from numpy.typing import NDArray

@dataclass(frozen=True)
class TangentPencil:
    coef: NDArray[Any]
    quad: NDArray[Any]
    linear: NDArray[Any]
    det: float
    chol: Tuple[NDArray[Any], bool]
    center: NDArray[Any]

def quad_matrix(coef: np.ndarray) -> np.ndarray: ...
def linear_vector(coef: np.ndarray) -> np.ndarray: ...
def build_tangent_pencil(mu: float, p: np.ndarray, q: np.ndarray) -> TangentPencil: ...
def target_prime_from_pencil(
    pencil: TangentPencil, p: np.ndarray, q: np.ndarray
) -> float: ...
def center_jacobian(pencil: TangentPencil) -> np.ndarray: ...
