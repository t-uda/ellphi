## Unreleased

### Changed

- `ellphi.grad.pdist_tangency_grad(...)` now dispatches the batched
  distance/gradient computation to the C++ backend when it is available
  (same auto-selection as `pdist_tangency`), giving orders-of-magnitude
  speedups for gradient-based workflows. The pure-Python implementation
  remains as the fallback and the public API is unchanged. Thanks to
  koki3070 and collaborators (TDA-ML) for the prototype and benchmarks.

## 0.1.2 - 2026-03-25

### Added

- Added the new `ellphi.grad` API for differentiable workflows:
  `tangency_grad(...)` for single-pair gradients,
  `pdist_tangency_grad(...)` for condensed pairwise distances plus a VJP
  pullback, and `coef_from_cov_grad(...)` for differentiable conversion from
  centres/covariances to packed conic coefficients.
- Added C++ backend build metadata helpers:
  `ellphi.build_info()`, `ellphi.cpp_linalg_kind()`, and
  `ellphi --build-info` / `python -m ellphi --build-info`.
- Added `EllipseCloud.from_cov(...)` for constructing clouds directly from
  precomputed centres/covariances, and `EllipseCloud.distance_matrix()` as a
  square-matrix convenience wrapper around `pdist_tangency`.

### Changed

- Source builds can now opt into an Eigen-based C++ linear-algebra backend via
  `ELLPHI_USE_EIGEN=1` (with `ELLPHI_EIGEN_INCLUDE` when needed). Build
  metadata now records the linear-algebra implementation and embedded backend
  version.

### Fixed

- `ellphi.visualization.ellipse_patch(...)` now preserves explicit
  `facecolor` / `fc` arguments while keeping hollow ellipses as the default.
- `coef_from_cov_grad(...)` now validates batch and dimensionality mismatches
  consistently and returns `NaN` outputs for singular covariances in line with
  `coef_from_cov(...)`.
- The C++ backend build cache is now invalidated when switching between the
  internal and Eigen linear-algebra modes, preventing stale extension reuse.

### Documentation

- Added a MkDocs documentation site with Getting Started, TDA workflow,
  differentiable tangency guide, API reference pages, and a notebook index
  that now includes a topology-optimisation example.

### Packaging

- Added Windows wheel build support (`win_amd64`, Python 3.10-3.12).
- Added release automation for TestPyPI/PyPI publishing via Trusted
  Publishing.
- Added Python 3.12 to the CI test matrix.

## 0.1.1 - 2025-12-14

### Added

- Full n-dimensional support for tangency solving across Python and C++ backends, including new conic
  packing helpers, EllipseCloud dimension tracking, and expanded differentiable gradient coverage with
  high-dimensional regression tests. (PR #31, #37, #38)
- A dedicated 3-D ellipsoid demo notebook plus documentation of the ndim extension review.
- Added the Algebraic Sigmoid Newton (`algsig+newton`) method. This provides an alternative unconstrained formulation for comparative studies and specific C++ backend use cases, though standard hybrid solvers remain recommended for general stability, especially in Python. (PR #54)
- Added support for Brent's hyperbolic method (`brenth`) as a robust alternative bracketing solver alongside the standard `brentq`. (PR #52)
- `scripts/hybrid_tuning.py` and the accompanying
  `docs/hybrid_tuning_summary.json` artifact summarising the empirical tuning
  run. The script generates extreme ellipse pairs across dimensions,
  benchmarks `ellphi.solver.tangency` for every requested backend (Python and
  C++ when available), and reports aggregate and per-dimension
  runtime/error/failure statistics in a Markdown table (with an opt-in plain
  view). Accuracy now relies on the relative tangency residual (rather than μ
  differences), and an optional `--plot-dir` flag emits log-log scatter plots
  of median time vs. error per backend. New options allow supplying or
  persisting case sets (`--cases-input/--cases-output`), overriding hybrid
  iteration pairs (`--hybrid-combos`), and prefixing plot names to keep
  multiple scenarios side by side, making it easy to reproduce and compare the
  parameter sweep backing the hybrid defaults. (PR #53)
- A new `ellphi` CLI entrypoint with a `--version` flag, and a runtime `ellphi.version_info()` helper
  to programmatically check the installed version. (PR #66)
- PEP 561 type marker (`py.typed`) and `.pyi` stub files are now included in the distribution,
  enabling better type checking and IDE autocompletion support. (PR #46)
- Added a test coverage badge to `README.md` to track code quality. (PR #77)
- Added comprehensive docstrings and documentation across the codebase to improve developer
  experience. (PR #76)

### Changed

- The `brentq+newton` hybrid now exposes configurable iteration counts all the
  way through the Python and C++ backends. Defaults are 28 Brent / 3 Newton iterations.
  Dispatch helpers, differentiable solvers, and tests now accept the new keyword arguments. (PR #80)
- Dropped support for Python 3.9. Python 3.10 or newer is now required. (PR #48)
- Updated `README.md` with a "Quick Start" section demonstrating basic tangency usage. (PR #61)

### Fixed

- The C++ backend now validates coefficient/point buffer lengths and clamps negative quadratic
  evaluations before taking square roots, while the Python wrapper rejects malformed coefficient arrays
  early.
- The Python backend now uses pivoted Gaussian elimination (via LU decomposition) as a robust fallback
  for singular pencils, matching the C++ backend's stability and preventing divergence in near-singular
  geometric configurations.

### Tooling

- AGENTS.md documents the exact CI command checklist and black[jupyter] is added so notebooks are
  formatted consistently.
- Expanded CI workflows to include documentation build checks, a dedicated test-build job, and
  dependency optimizations for faster execution. (PR #70, #71, #72)
