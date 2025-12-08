# Python/NumPy/SciPy Version Stability Investigation

## Background
Recent CI failures surfaced after adding a very strict regression case for the Python backend. The test passed locally but diverged on CI when changing Python (and implicitly NumPy/SciPy) versions. The goal of this investigation was to align our local environment with CI, pinpoint the numerical instability, and document a mitigation strategy together with future CI hardening steps.

## CI environment summary
- GitHub Actions currently exercises Python **3.10** and **3.11** for linting and tests, with type-checking limited to Python **3.11**.【F:.github/workflows/python-app.yml†L13-L47】
- Dependency versions are pinned by `poetry.lock`. The current lock resolves to **NumPy 2.2.6** and **SciPy 1.15.3**, which are the wheels installed in CI for both Python 3.10 and 3.11.【F:poetry.lock†L2112-L2134】【F:poetry.lock†L3436-L3453】

## Reproduction attempts
- A dedicated Python 3.11 virtual environment was created to mirror CI, but package installation was blocked by the sandbox’s lack of access to `files.pythonhosted.org`, so NumPy/SciPy wheels for cp311 could not be fetched. (See terminal log for the connectivity failure.)【1d01bd†L1-L10】
- All regression tests were executed under Python 3.12 with the locked NumPy/SciPy versions to establish a clean baseline; they passed (122 tests in ~3.3s).【81c360†L1-L3】 Although this does not perfectly mirror CI, it validated the fix below against the newest available interpreter in this environment.

## Root cause analysis
- The Python backend computed conic centres via Cholesky factorisation with a fallback to `numpy.linalg.lstsq` when factorisation failed. The C++ backend, however, falls back to an explicit Gaussian-elimination solver with partial pivoting.
- `numpy.linalg.lstsq` delegates to LAPACK routines whose pivoting/regularisation changed between NumPy releases, making the Python backend sensitive to minor BLAS/LAPACK differences even when the high-level algorithm is unchanged. In near-degenerate pencil cases, this produced small residuals that altered the Newton search direction enough to trigger divergence on some interpreter/library combinations.

## Fix implemented
- Introduced a C++-style Gaussian-elimination solver in `_solver_python` and swapped the `lstsq` fallback for this deterministic path. This keeps the Python backend numerically aligned with the C++ implementation across NumPy/SciPy versions.【F:src/ellphi/_solver_python.py†L70-L117】【F:src/ellphi/_solver_python.py†L130-L141】
- Added a regression test that forces the Cholesky factorisation to fail on an indefinite 2×2 matrix and verifies the fallback produces the expected solution. This anchors the behaviour to the C++ strategy and guards against future regressions when dependencies change.【F:tests/test_numerical_stability.py†L6-L11】【F:tests/test_numerical_stability.py†L63-L73】

## Recommendations for future CI hardening
- **Broaden the version matrix**: add a third Python (e.g., 3.12) and at least one older NumPy/SciPy pair (e.g., NumPy 1.26/SciPy 1.12) to detect LAPACK-sensitive regressions earlier. A nightly or scheduled workflow can keep the main CI fast while still giving coverage.
- **Document non-recommended ranges**: if further testing shows divergence limited to specific NumPy/SciPy builds, explicitly call those out in the README and/or raise installation warnings to steer users away from unstable combinations.
- **Deterministic solver preference**: keep preferring explicitly pivoted linear solvers (as now implemented) over `lstsq` when the system is square, to minimise dependency-specific variation.
- **Offline mirroring**: cache wheels for the CI-supported Python versions in the development container to make it easier to reproduce CI behaviour when external downloads are restricted.

These steps, combined with the solver alignment implemented here, should reduce sensitivity to subtle numeric differences across interpreter and LAPACK releases.
