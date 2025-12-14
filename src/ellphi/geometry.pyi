from ellphi import FloatArray

__all__ = [
    "unit_vector",
    "axes_from_cov",
    "coef_from_axes",
    "coef_from_cov",
    "pack_conic",
    "unpack_conic",
    "unpack_single_conic",
    "infer_dim_from_coef_length",
]

def unit_vector(theta: float) -> FloatArray: ...
def axes_from_cov(cov: FloatArray, /, *, scale: float = 1.0): ...
def infer_dim_from_coef_length(length: int) -> int: ...
def pack_conic(
    matrices: FloatArray, linear: FloatArray, constant: FloatArray
) -> FloatArray: ...
def unpack_conic(
    coef: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]: ...
def unpack_single_conic(
    coef: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]: ...
def coef_from_axes(X: float, r0: float, r1: float, theta: float) -> FloatArray: ...
def coef_from_cov(
    X: FloatArray, cov: FloatArray, /, *, scale: float = 1.0
) -> FloatArray: ...
