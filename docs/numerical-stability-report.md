# Numerical Stability Investigation (Python Backends)

## CI and Dependency Baseline
- CI runs on Ubuntu with Python 3.10 and 3.11, installing dependencies via Poetry and running flake8, black, pytest, mypy, and stubtest. 【F:.github/workflows/python-app.yml†L6-L46】
- The Poetry lockfile currently resolves NumPy to 2.2.6 and SciPy to 1.15.3 for all supported interpreters (>=3.10). 【F:poetry.lock†L2111-L2174】【F:poetry.lock†L3435-L3498】

## Findings
- The divergent 5D test case (`tests/test_numerical_stability.py::test_divergent_algsig_newton_case`) succeeds on the C++ backend because its fallback solver performs Gaussian elimination with pivoting when Cholesky factorization fails. 【F:tests/test_numerical_stability.py†L5-L43】【F:src/ellphi/_tangency_cpp_impl.cpp†L300-L347】
- The Python backend previously fell back to `numpy.linalg.lstsq`, which can follow a different numerical path depending on NumPy/LAPACK versions, leading to non-convergence on Python 3.11 with newer NumPy/SciPy.
- Aligning the Python fallback with the C++ elimination path removes this source of version-dependent behavior and better matches the intended algorithm.

## Changes Implemented
- Added a Gaussian elimination solver with partial pivoting to the Python backend and swapped the fallback in `_center` to use it instead of `lstsq`, mirroring the C++ code path. 【F:src/ellphi/_solver_python.py†L38-L85】【F:src/ellphi/_solver_python.py†L125-L140】

## Remaining Risk and Recommendations
- Network restrictions in this environment prevented installing dependencies, so CI-equivalent test runs (black/flake8/pytest/mypy/stubtest) could not be executed here. Future validation should confirm the fix across Python 3.10/3.11 with NumPy 2.2.6 and SciPy 1.15.3. 【F:AGENTS.md†L15-L74】
- To guard against similar regressions, consider expanding CI to a matrix over critical NumPy/SciPy versions (e.g., latest and minimum supported) alongside Python versions.
- If further instability is observed, document any non-supported NumPy/SciPy releases and pin or exclude them in `pyproject.toml` along with release notes.
