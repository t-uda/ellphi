# Notebook Examples

The notebooks below demonstrate EllPHi's key workflows end-to-end.
They are hosted alongside the source code on GitHub and can be viewed
directly in the browser or cloned and run locally.

!!! note "Future plans"
    The notebook collection is planned to move to a dedicated repository
    with interactive [Marimo](https://marimo.io) versions that run
    entirely in the browser via WebAssembly — no local installation
    required.  Links on this page will be updated when that migration
    is complete.

## Getting started

| Notebook | Description |
|---|---|
| [Quick Start](https://github.com/t-uda/ellphi/blob/main/notebooks/quickstart.ipynb) | Install EllPHi, build an ellipse cloud, and compute pairwise tangency distances in a few lines. |

## Persistent homology workflows

| Notebook | Description |
|---|---|
| [Anisotropy-aware PH — 6 rings](https://github.com/t-uda/ellphi/blob/main/notebooks/eph-6rings-PH.ipynb) | Full pipeline from a six-ring point cloud to a persistence diagram using elliptic tangency distances. |
| [PH figures — standard vs elliptic](https://github.com/t-uda/ellphi/blob/main/notebooks/eph-6rings-PH-figures.ipynb) | Side-by-side comparison of standard (Euclidean) and anisotropic (elliptic) persistent homology on the same data. |
| [n-dimensional demo (3D)](https://github.com/t-uda/ellphi/blob/main/notebooks/ndim-demo-3d.ipynb) | Extends the quick-start to a 3-D point cloud, showing that the solver and distance pipeline work in any dimension. |

## Differentiable solver

| Notebook | Description |
|---|---|
| [Topology Optimisation with Gradients](https://github.com/t-uda/ellphi/blob/main/notebooks/topology_optimization.ipynb) | Gradient-based optimisation of ellipsoid configurations using `ellphi.grad` and the VJP interface. *(Link active after next release.)* |

## Benchmarks

| Notebook | Description |
|---|---|
| [Performance Benchmark](https://github.com/t-uda/ellphi/blob/main/notebooks/performance_benchmark.ipynb) | Compares per-call timings of `pdist_tangency` across the pure-Python and compiled C++ backends. |
