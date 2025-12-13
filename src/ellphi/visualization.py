"""
ellphi.visualization  –  visualization helpers for ellipse cloud
================================================================

"""

import numpy
import matplotlib.pyplot as plt
from .geometry import axes_from_cov

__all__ = ["ellipse_patch"]


def ellipse_patch(X, r_major=1, r_minor=1, theta=0, *, cov=None, scale=1, **kwgs):
    """Creates a matplotlib `Ellipse` patch.

    This function can create an ellipse patch from either the major and minor
    axes and angle, or from a covariance matrix.

    Args:
        X: The center of the ellipse.
        r_major: The major radius of the ellipse.
        r_minor: The minor radius of the ellipse.
        theta: The angle of the major axis in radians.
        cov: An optional covariance matrix. If provided, `r_major`,
            `r_minor`, and `theta` will be computed from it.
        scale: An optional scaling factor for the ellipse.
        **kwgs: Additional keyword arguments to pass to the `Ellipse` patch.

    Returns:
        A `matplotlib.patches.Ellipse` object.
    """
    if cov is not None:
        r_major, r_minor, theta = axes_from_cov(cov)
    ellipse = plt.matplotlib.patches.Ellipse(
        X,
        width=2 * r_major * scale,
        height=2 * r_minor * scale,
        angle=numpy.degrees(theta),
        facecolor="none",
        **kwgs,
    )
    return ellipse
