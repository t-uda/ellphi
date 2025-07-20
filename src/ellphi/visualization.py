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

# def ellipse_curve_cov(x0=0, y0=0, cov=[[1,0],[0,1]], scale=1, N=100):
#     """
#     入力：
#     中心 (x0, y0)
#     共分散行列 cov
#     媒介変数空間分割数 N
#     出力：
#     楕円の媒介変数表示 {'x': x, 'y': y}（Plotly plot に対応）
#     """
#     D, U = np.linalg.eig(cov)
#     sqrtD = np.diag(np.sqrt(D))
#     # 媒介変数
#     t = np.linspace(0, 2 * np.pi, N)
#     # 単位円→楕円のスケール回転平行移動変換を用いた媒介変数表示
#     c = np.transpose([[x0, y0]])
#     x, y = U @ (scale * sqrtD @ unit_vector(t)) + c
#     return {'x': x, 'y': y}

def ellipse_patch(x0, y0, r1=1, r2=1, theta=0, *, cov=None, scale=1):
    if cov is not None:
        r1, r2, theta = axes_from_cov(cov)
    ellipse = plt.matplotlib.patches.Ellipse(
        (x0, y0),
        width=2*r1*scale,
        height=2*r2*scale,
        angle=numpy.degrees(theta),
        edgecolor='black',
        facecolor='none',
        linestyle='-'
    )
    return ellipse

