import numpy as np
from .geometry import unpack_single_conic as unpack_single_conic
from dataclasses import dataclass

@dataclass(frozen=True)
class TangentPencil:
    coef: np.ndarray
    quad: np.ndarray
    linear: np.ndarray
    det: float
    inv_quad: np.ndarray
    center: np.ndarray

def quad_matrix(coef: np.ndarray) -> np.ndarray: ...
def linear_vector(coef: np.ndarray) -> np.ndarray: ...
def build_tangent_pencil(mu: float, p: np.ndarray, q: np.ndarray) -> TangentPencil: ...
def target_prime_from_pencil(
    pencil: TangentPencil, p: np.ndarray, q: np.ndarray
) -> float: ...
def center_jacobian(pencil: TangentPencil) -> np.ndarray: ...
