## Unreleased

No unreleased changes.

## 0.1.1 - 2025-12-01

### Added

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
  parameter sweep backing the hybrid defaults.
- Full n-dimensional support for tangency solving across Python and C++ backends, including new conic
  packing helpers, EllipseCloud dimension tracking, and expanded differentiable gradient coverage with
  high-dimensional regression tests.
- A dedicated 3-D ellipsoid demo notebook plus documentation of the ndim extension review.
- A new `ellphi` CLI entrypoint with a `--version` flag, and a runtime `ellphi.version_info()` helper
  to programmatically check the installed version.
- PEP 561 type marker (`py.typed`) and `.pyi` stub files are now included in the distribution,
  enabling better type checking and IDE autocompletion support.

### Changed

- The `brentq+newton` hybrid now exposes configurable iteration counts all the
  way through the Python and C++ backends. Defaults are dimension aware: the
  historical 2D tuning (8 Brent / 3 Newton iterations) is retained, while n>2
  problems use the empirically tuned 28 / 6 budget. Dispatch helpers,
  differentiable solvers, and tests now accept the new keyword arguments.
- Dropped support for Python 3.9. Python 3.10 or newer is now required.
- Updated `README.md` with a "Quick Start" section demonstrating basic tangency usage.

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
