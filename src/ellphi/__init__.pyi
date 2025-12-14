from numpy.typing import NDArray
import numpy as np

FloatArray = NDArray[np.float64]

from .ellcloud import (
    EllipseCloud as EllipseCloud,
    LocalCov as LocalCov,
    ellipse_cloud as ellipse_cloud,
)
from .geometry import (
    axes_from_cov as axes_from_cov,
    coef_from_axes as coef_from_axes,
    coef_from_cov as coef_from_cov,
    unit_vector as unit_vector,
)
from .solver import (
    TangencyResult as TangencyResult,
    has_cpp_backend as has_cpp_backend,
    pdist_tangency as pdist_tangency,
    pencil as pencil,
    quad_eval as quad_eval,
    tangency as tangency,
)
from ._version import __version__ as __version__

def version_info() -> str: ...
def _main() -> None: ...

__all__ = [
    "unit_vector",
    "axes_from_cov",
    "coef_from_axes",
    "coef_from_cov",
    "ellipse_cloud",
    "EllipseCloud",
    "LocalCov",
    "quad_eval",
    "pencil",
    "tangency",
    "pdist_tangency",
    "TangencyResult",
    "has_cpp_backend",
    "FloatArray",
    "__version__",
    "version_info",
]
