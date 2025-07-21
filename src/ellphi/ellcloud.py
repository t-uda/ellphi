
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence, Iterator, Optional

import numpy
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform, pdist

from .geometry import axes_from_cov, coef_from_cov
from .solver import pdist_tangency

__all__ = [
    "ellipse_cloud",
    "EllipseCloud",
    "LocalCov"
]


@dataclass
class EllipseCloud:
    """Container for an ellipse cloud with convenience methods."""
    coef: numpy.ndarray  # (N, 6)
    mean: numpy.ndarray  # (N, 2)
    cov:  numpy.ndarray  # (N, 2, 2)
    k:    int
    nbd:  numpy.ndarray  # (N, k)  k-NN indices
    n:    int = field(init=False)

    # ---- automatic field from coef.shape ---------------------------------
    def __post_init__(self):
        self.n = self.coef.shape[0]

    # ---- basic Python protocol ------------------------------------------
    def __len__(self) -> int:
        return self.n

    def __iter__(self) -> Iterator[numpy.ndarray]:
        return iter(self.coef)

    def __getitem__(self, idx) -> EllipseCloud:
        """Return a *view* (not copy) subset as a new EllipseCloud."""
        return self.coef[idx]

    def __str__(self):
        coef_str = f"coef=array<{self.coef.shape}>"
        mean_str = f"mean=array<{self.mean.shape}>"
        cov_str = f"cov=array<{self.cov.shape}>"
        k_str = f"k={self.k}"
        nbd_str = f"nbd=array<{self.nbd.shape}>"
        param_str = ', '.join([
            coef_str, mean_str, cov_str, k_str, nbd_str
        ])
        return f"EllipseCloud({param_str})"

    # ---- visualisation ---------------------------------------------------
    def plot(
        self,
        ids: Optional[Sequence[int]] = None,
        ax: Optional[plt.Axes] = None,
        scale: float = 1.0,
        # facecolor: str = "none",
        # edgecolor: str = "C0",
        # alpha: float = 0.8,
        **kwgs,
    ) -> plt.Axes:
        """
        Quick matplotlib visualisation.

        Parameters
        ----------
        ids
            Subset of ellipse indices to draw.  None = all.
        ax
            Existing axes; if None, creates a new figure.
        """
        from .visualization import ellipse_patch
        if ax is None:
            fig, ax = plt.subplots()

        ids = numpy.arange(self.n) if ids is None else numpy.asarray(ids)
        axes = axes_from_cov(self.cov[ids])
        for i, r_major, r_minor, theta in zip(ids, *axes):
            ellpatch = ellipse_patch(
                    self.mean[i], r_major, r_minor, theta,
                    scale=scale, **kwgs)
            ax.add_patch(ellpatch)
        return ax

    def pdist_tangency(self):
        return pdist_tangency(self)

    @classmethod
    def from_point_cloud(
            cls: EllipseCloud,
            X: numpy.ndarray,
            *,
            method="local_cov",
            rescaling="none",
            **kwgs) -> EllipseCloud:
        if method == "local_cov":
            ellcloud = cls.from_local_cov(X, **kwgs)
        else:
            raise NotImplementedError(
                f"Unknown method '{method}':\n" +
                "The supported method is 'local_cov'."
            )
        if rescaling != "none":
            ellcloud.rescale(method=rescaling)
        return ellcloud

    @classmethod
    def from_local_cov(
            cls: EllipseCloud, X: numpy.ndarray, *, k: int = 5
            ) -> EllipseCloud:
        return LocalCov(k=k)(X)

    def rescale(self, *, method="median") -> float:
        scales = numpy.sqrt(numpy.linalg.eigh(self.cov).eigenvalues)
        if method == "median":
            ell_scales = numpy.median(scales, axis=0)
        elif method == "average":
            ell_scales = numpy.average(scales, axis=0)
        else:
            raise NotImplementedError(
                f"Unknown method '{method}':\n" +
                "The supported method is 'median' or 'average'."
            )
        ell_scale = ell_scales[1]**2 / ell_scales[0]
        self.cov /= ell_scale**2
        self.coef *= ell_scale**2
        return ell_scale


# alias
ellipse_cloud = EllipseCloud.from_point_cloud


@dataclass(frozen=True)
class LocalCov:
    """Algorithm creating Ellipse Cloud from k-nearest neighbours."""
    k: int = 5  # 近傍点数

    # 将来オプションが増えても dataclass なので拡張しやすい
    # 例: weight_func: Literal["uniform", "distance"]

    # main entry: make EllipseCloud from raw Nx2 points -----------------
    def __call__(self, X: numpy.ndarray) -> EllipseCloud:
        """
        Parameters
        ----------
        X : ndarray, shape (N, 2)
            入力点群（x,y）

        Returns
        -------
        EllipseCloud
            centres, covs, coeffs などを含むクラウドオブジェクト
        """
        k = self.k
        d = squareform(pdist(X))  # Euclidean distance matrix
        # argsort したものから :near だけとると重複が生じるので削る
        near_subsets = numpy.unique(numpy.argsort(d, axis=1)[:, :k], axis=0)
        # 各サブセットをソートしてタプルに変換
        sorted_subsets = [tuple(sorted(subset)) for subset in near_subsets]
        unique_subsets = numpy.unique(sorted_subsets, axis=0)  # 重複を取り除く
        knbd = X[unique_subsets]
        means = numpy.mean(knbd, axis=1)
        rel_nbd = knbd - means[:, None, :]
        covs = rel_nbd.transpose(0, 2, 1) @ rel_nbd / (k - 1)
        coefs = coef_from_cov(means, covs)
        return EllipseCloud(coefs, means, covs, k, unique_subsets)
