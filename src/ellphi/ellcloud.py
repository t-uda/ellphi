"""
ellphi.ellcloud  –  ellipse cloud interfaces
=============================================================

- ellipse_cloud(X, method="local_cov", rescaling="none", k=5)
- class EllipseCloud
- class LocalCov

"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence, Iterator

import numpy
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform, pdist

from .geometry import axes_from_cov, coef_from_cov, infer_dim_from_coef_length
from .solver import pdist_tangency

__all__ = ["ellipse_cloud", "EllipseCloud", "LocalCov"]


@dataclass
class EllipseCloud:
    """A container for an ellipse cloud with convenience methods.

    This class provides a convenient way to store and manipulate a collection of
    ellipses. It includes methods for plotting, computing pairwise tangency
    distances, and creating an ellipse cloud from a point cloud.

    Attributes:
        coef: A NumPy array of shape `(N, m)` containing the conic coefficients
            for each ellipse.
        mean: A NumPy array of shape `(N, d)` containing the mean of each
            ellipse.
        cov: A NumPy array of shape `(N, d, d)` containing the covariance
            matrix of each ellipse.
        k: The number of nearest neighbors used to create the ellipse cloud.
        nbd: A NumPy array of shape `(N, k)` containing the indices of the
            k-nearest neighbors for each ellipse.
        n: The number of ellipses in the cloud.
        n_dim: The number of dimensions of the ellipses.
    """

    coef: numpy.ndarray  # (N, m)
    mean: numpy.ndarray  # (N, d)
    cov: numpy.ndarray  # (N, d, d)
    k: int
    nbd: numpy.ndarray  # (N, k)  k-NN indices
    n: int = field(init=False)
    n_dim: int = field(init=False)

    # ---- automatic field from coef.shape ---------------------------------
    def __post_init__(self):
        if self.coef.ndim != 2:
            raise ValueError("Coefficient array must be two-dimensional")
        self.n = self.coef.shape[0]
        coef_length = self.coef.shape[1]
        self.n_dim = infer_dim_from_coef_length(coef_length)
        expected_mean_shape = (self.n, self.n_dim)
        expected_cov_shape = (self.n, self.n_dim, self.n_dim)
        if self.mean.shape != expected_mean_shape:
            raise ValueError(
                "Mean array has shape "
                f"{self.mean.shape}, expected {expected_mean_shape}"
            )
        if self.cov.shape != expected_cov_shape:
            raise ValueError(
                "Covariance array has shape "
                f"{self.cov.shape}, expected {expected_cov_shape}"
            )
        if self.nbd.shape and self.nbd.shape[0] != self.n:
            raise ValueError(
                "Neighbourhood index array must have first dimension "
                f"{self.n}, got {self.nbd.shape}"
            )

    # ---- basic Python protocol ------------------------------------------
    def __len__(self) -> int:
        return self.n

    def __iter__(self) -> Iterator[numpy.ndarray]:
        return iter(self.coef)

    def __getitem__(self, idx) -> numpy.ndarray:
        """Returns the conic coefficient array for a single ellipsoid.

        Args:
            idx: The index of the ellipsoid to retrieve.

        Returns:
            A NumPy array representing the conic coefficients of the specified
            ellipsoid.
        """
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
        ids: Sequence[int] | None = None,
        ax: plt.Axes | None = None,
        scale: float = 1.0,
        # facecolor: str = "none",
        # edgecolor: str = "C0",
        # alpha: float = 0.8,
        **kwgs,
    ) -> plt.Axes:
        """A quick matplotlib visualization of the ellipse cloud.

        This method provides a simple way to visualize the ellipse cloud using
        matplotlib. It can plot all ellipses or a subset of them, and it can
        be used with an existing `Axes` object or create a new one.

        Args:
            ids: A sequence of indices of the ellipses to plot. If `None`, all
                ellipses are plotted.
            ax: An existing `Axes` object to plot on. If `None`, a new
                figure and `Axes` are created.
            scale: A scaling factor for the ellipses.
            **kwgs: Additional keyword arguments to pass to the
                `ellipse_patch` function.

        Returns:
            The `Axes` object with the plotted ellipses.
        """
        from .visualization import ellipse_patch

        if self.n_dim != 2:
            raise NotImplementedError("Plotting is only supported for 2D ellipses")

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
        """Computes pairwise tangency distances for the ellipse cloud.

        This is a convenience method that calls `ellphi.solver.pdist_tangency`.

        Args:
            parallel: If `True`, the computation is performed in parallel.
            n_jobs: The number of jobs to run in parallel. See
                `ellphi.solver.pdist_tangency` for more details.
            backend: The backend to use for the computation. Can be one of
                "auto", "python", or "cpp".

        Returns:
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
        """Creates an `EllipseCloud` from a point cloud.

        This class method provides a way to create an `EllipseCloud` from a
        given point cloud. It supports different methods for creating the
        ellipses and for rescaling them.

        Args:
            X: A NumPy array of shape `(N, d)` representing the input point
                cloud.
            method: The method to use for creating the ellipses. Currently,
                only "local_cov" is supported.
            rescaling: The method to use for rescaling the ellipses. Can be
                one of "none", "median", or "average".
            **kwgs: Additional keyword arguments to pass to the specified
                `method`.

        Returns:
            An `EllipseCloud` object created from the point cloud.
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
        """Creates an `EllipseCloud` from a point cloud using local covariance.

        This class method creates an `EllipseCloud` from a point cloud by
        computing the local covariance of the k-nearest neighbors for each
        point.

        Args:
            X: A NumPy array of shape `(N, d)` representing the input point
                cloud.
            k: The number of nearest neighbors to use for computing the local
                covariance.

        Returns:
            An `EllipseCloud` object created from the point cloud.
        """
        return LocalCov(k=k)(X)

    def rescale(self, *, method="median") -> float:
        """Applies rescaling to all the ellipses in the cloud.

        This method rescales all the ellipses in the cloud based on the
        specified method.

        Args:
            method: The rescaling method to use. Can be "median" or "average".

        Returns:
            The scaling factor used to rescale the ellipses.
        """
        if self.n_dim != 2:
            raise NotImplementedError(
                "Rescaling is currently implemented for 2D ellipses only"
            )
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
    """An algorithm for creating an `EllipseCloud` from k-nearest neighbors.

    This class implements an algorithm for creating an `EllipseCloud` from a
    point cloud by computing the local covariance of the k-nearest neighbors
    for each point.

    Attributes:
        k: The number of nearest neighbors to use for computing the local
            covariance.
    """

    k: int = 5

    # 将来オプションが増えても dataclass なので拡張しやすい
    # 例: weight_func: Literal["uniform", "distance"]

    # main entry: make EllipseCloud from raw NxD points -----------------
    def __call__(self, X: numpy.ndarray) -> EllipseCloud:
        """Creates an `EllipseCloud` from a point cloud.

        This method takes a point cloud and creates an `EllipseCloud` by
        computing the local covariance of the k-nearest neighbors for each
        point.

        Args:
            X: A NumPy array of shape `(N, d)` representing the input point
                cloud.

        Returns:
            An `EllipseCloud` object created from the point cloud. Note that
            the number of ellipses can be less than the number of input points
            `N` because point subsets with identical k-NN are merged into a
            single ellipse.
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
