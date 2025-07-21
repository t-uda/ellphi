"""
ellphi.visualization  –  visualization helpers for ellipse cloud
================================================================

"""

import numpy
import matplotlib.pyplot as plt
from .geometry import axes_from_cov

__all__ = [
    "ellipse_patch"
]


def ellipse_patch(X, r_major=1, r_minor=1, theta=0, *, cov=None, scale=1,
                  **kwgs):
    if cov is not None:
        r_major, r_minor, theta = axes_from_cov(cov)
    ellipse = plt.matplotlib.patches.Ellipse(
        X,
        width=2 * r_major * scale,
        height=2 * r_minor * scale,
        angle=numpy.degrees(theta),
        facecolor='none',
        **kwgs
    )
    return ellipse
