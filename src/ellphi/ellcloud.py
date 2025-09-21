"""
ellphi.ellcloud  –  ellipse cloud interfaces
=============================================================

- ellipse_cloud(X, method="local_cov", rescaling="none", k=5)
- class EllipseCloud
- class LocalCov

"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence, Iterator, Optional

import numpy
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform, pdist

from .geometry import axes_from_cov, coef_from_cov
from .solver import pdist_tangency

__all__ = ["ellipse_cloud", "EllipseCloud", "LocalCov"]


@dataclass
class EllipseCloud:
    """Container for an ellipse cloud with convenience methods."""

    coef: numpy.ndarray  # (N, 6)
    mean: numpy.ndarray  # (N, 2)
    cov: numpy.ndarray  # (N, 2, 2)
    k: int
    nbd: numpy.ndarray  # (N, k)  k-NN indices
    n: int = field(init=False)

    # ---- automatic field from coef.shape ---------------------------------
    def __post_init__(self):
        self.n = self.coef.shape[0]

    # ---- basic Python protocol ------------------------------------------
    def __len__(self) -> int:
        return self.n

    def __iter__(self) -> Iterator[numpy.ndarray]:
        return iter(self.coef)

    def __getitem__(self, idx) -> numpy.ndarray:
        """Return the conic coefficient array (6,) for a single ellipse."""
        return self.coef[idx]

    def __str__(self):
        coef_str = f"coef=array<{self.coef.shape}>"
        mean_str = f"mean=array<{self.mean.shape}>"
        cov_str = f"cov=array<{self.cov.shape}>"
        k_str = f"k={self.k}"
        nbd_str = f"nbd=array<{self.nbd.shape}>"
        param_str = ", ".join([coef_str, mean_str, cov_str, k_str, nbd_str])
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

        idarr = numpy.arange(self.n) if ids is None else numpy.asarray(ids)
        axes = axes_from_cov(self.cov[idarr])
        for i, r_major, r_minor, theta in zip(idarr, *axes):
            ellpatch = ellipse_patch(
                self.mean[i], r_major, r_minor, theta, scale=scale, **kwgs
            )
            ax.add_patch(ellpatch)
        return ax

    def pdist_tangency(
        self,
        *,
        parallel: bool = True,
        n_jobs: int | None = -1,
        backend: str = "auto",
    ):
        """
        Compute pairwise tangency distances for the ellipse cloud.

        This is a convenience method that calls `ellphi.solver.pdist_tangency`.

        Parameters
        ----------
        parallel : bool, optional
            If True (default), compute the tangencies in parallel.
        n_jobs : int or None, optional
            Number of jobs to run in parallel. See `ellphi.solver.pdist_tangency`.
        backend : {"auto", "python", "cpp"}
            Backend used for the tangency computation.

        Returns
        -------
        numpy.ndarray
            A condensed distance matrix of tangency distances.
        """
        return pdist_tangency(
            self,
            parallel=parallel,
            n_jobs=n_jobs,
            backend=backend,
        )

    @classmethod
    def from_point_cloud(
        cls: type[EllipseCloud],
        X: numpy.ndarray,
        *,
        method="local_cov",
        rescaling="none",
        **kwgs,
    ) -> EllipseCloud:
        """
        Parameters
        ----------
        X : ndarray, shape (N, 2)
            Input point cloud (x, y)
        method : str
            Conversion algorithm. The supported method is "local_cov".
        rescaling : str
            The supported rescaling method is "none", "median", or "average".
            See also `EllipseCloud.rescale`
        **kwgs :
            Passed to the corresponding algorithm specified by `method`.

        Returns
        -------
        EllipseCloud
            Resulting ellipse cloud constructed by `method`
        """
        if method == "local_cov":
            ellcloud = cls.from_local_cov(X, **kwgs)
        else:
            raise NotImplementedError(
                f"Unknown method '{method}':\n" + "The supported method is 'local_cov'."
            )
        if rescaling != "none":
            ellcloud.rescale(method=rescaling)
        return ellcloud

    @classmethod
    def from_local_cov(
        cls: type[EllipseCloud], X: numpy.ndarray, *, k: int = 5
    ) -> EllipseCloud:
        return LocalCov(k=k)(X)

    def rescale(self, *, method="median") -> float:
        """
        Apply rescaling to all the ellipses.
        The supported method is "median" or "average".
        """
        eigvals = numpy.linalg.eigvalsh(self.cov)
        scales = numpy.sqrt(eigvals)
        if method == "median":
            ell_scales = numpy.median(scales, axis=0)
        elif method == "average":
            ell_scales = numpy.average(scales, axis=0)
        else:
            raise NotImplementedError(
                f"Unknown method '{method}':\n"
                + "The supported method is 'median' or 'average'."
            )
        ell_scale = ell_scales[1] ** 2 / ell_scales[0]
        self.cov /= ell_scale**2
        self.coef *= ell_scale**2
        return float(ell_scale)


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
            Input point cloud (x, y)

        Returns
        -------
        EllipseCloud
            Resulting ellipse cloud constructed by local covariance.
            Note that the number of ellipses can be less than the number of
            input points `N` because point subsets with identical k-NN are
            merged into a single ellipse.
        """
        k = self.k
        if k < 2:
            raise ValueError(
                "Local covariance requires at least two neighbours (k >= 2); "
                f"got k={k}."
            )
        d = squareform(pdist(X))  # Euclidean distance matrix
        neighbour_indices = numpy.argsort(d, axis=1)[:, :k]
        sorted_subsets = numpy.sort(neighbour_indices, axis=1)
        unique_subsets = numpy.unique(sorted_subsets, axis=0)
        knbd = X[unique_subsets]
        means = numpy.mean(knbd, axis=1)
        rel_nbd = knbd - means[:, None, :]
        nbd_size = knbd.shape[1]
        if nbd_size < 2:
            raise ValueError(
                "Local covariance requires neighbourhoods with at least two "
                f"points; got {nbd_size}."
            )
        covs = rel_nbd.transpose(0, 2, 1) @ rel_nbd / (nbd_size - 1)
        coefs = coef_from_cov(means, covs)
        return EllipseCloud(coefs, means, covs, k, unique_subsets)
