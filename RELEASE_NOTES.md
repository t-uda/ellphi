## Unreleased

### CI

- Added Windows wheel build workflow (`win_amd64`, Python 3.10–3.12).
- Added release workflow for PyPI publishing via Trusted Publishing.
- Added Python 3.12 to the CI test matrix.

### Documentation

- Expanded MkDocs site: added Notebook Examples page to User Guide, listing
  all six notebooks grouped by theme with links to the GitHub viewer and a
  note on the planned migration to an interactive Marimo-based repository.
- Added EllPHi logo to the navbar, favicon, and index page; root
  `ellphi-logo.png` replaced with a transparency-processed version for
  clean rendering in GitHub dark mode.
- Unified mathematical vector notation throughout the docs: vectors now use
  `\bm{·}` (bold-italic via `\boldsymbol`), matrices use plain capitals,
  scalars use plain italic.  `\bm` is defined as a MathJax macro in
  `docs/javascripts/mathjax.js`.
- Fixed markdown and MathJax formatting in Design Notes: resolved multi-line
  inline-math parsing failures in `algebraic_sigmoid_newton.md`, converted
  Unicode math notation in `hybrid_tuning.md` to LaTeX, and replaced a
  heredoc split across bullet points with a proper code block in
  `eigen_backend_plan.md`.
- Applied English proofreading corrections across User Guide pages
  (terminology, punctuation, British spelling, precision of mathematical
  prose).
- Introduced `CLAUDE.md` for Claude Code-specific guidelines (subagent
  permissions, orchestration style); added documentation conventions
  (build workflow, notation, English style) to `CONTRIBUTING.md` and
  `AGENTS.md`.

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
