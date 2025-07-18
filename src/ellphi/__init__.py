"""
ellphi: fast ellipse tangency solver
"""

from importlib.metadata import version as _version

from .geometry import Ellipse, ellipse_coeffs, coeff_matrix
from .solver import tangency, pairwise_tangency, TangencyResult

__all__ = [
    "Ellipse",
    "ellipse_coeffs",
    "coeff_matrix",
    "tangency",
    "pairwise_tangency",
    "TangencyResult",
]

__version__ = _version(__name__)

