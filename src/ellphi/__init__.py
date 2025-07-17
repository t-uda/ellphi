"""
ellphi: fast ellipse tangency solver
"""

from importlib.metadata import version as _version
from .epencil import *

__all__ = [
        "pencil",
        "target_function", "target_function_prime",
        "find_intersect", "find_intersect_mu"
        ]
__version__ = _version(__name__)

