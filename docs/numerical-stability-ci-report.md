# Numerical stability regression under Python 3.11 CI

## Background
A newly added stress case (`tests/test_numerical_stability.py::test_divergent_algsig_newton_case`) converges locally but diverged on the CI Python 3.11 job. The CI matrix currently targets Ubuntu runners with Python 3.10 and 3.11, installing dependencies from `poetry.lock`. The lock file resolves to NumPy 2.2.6 and SciPy 1.15.3, so the divergent behavior pointed to version-specific linear algebra differences instead of test data errors.

## CI environment snapshot
- GitHub Actions matrix: Ubuntu, Python 3.10 and 3.11, Poetry install with dev extras. 
- Locked dependencies relevant to the solver: NumPy 2.2.6, SciPy 1.15.3.

These versions mean both CI jobs share the same LAPACK-backed numerical routines; however, Python 3.11 pulls the most recent manylinux wheels where subtle BLAS/LAPACK changes (e.g., QR pivoting in `numpy.linalg.lstsq`) can shift convergence for ill-conditioned systems.

## Findings
- The Python backend computed conic centers via `numpy.linalg.lstsq` after a failed Cholesky factorization. On older NumPy/SciPy builds this aligned with the C++ backend, but newer wheels showed larger residuals for the stress case, perturbing Newton's direction enough to fail convergence.
- The C++ backend never uses least-squares; it falls back to partial-pivot Gaussian elimination when Cholesky fails. The discrepancy explains why C++ continued to converge while Python diverged under the same coefficients.
- Local reproduction inside this sandbox used Python 3.12 (outside the CI matrix) and hit a segmentation fault in the C++ extension before reaching the stress case, so the analysis below relies on code inspection rather than identical CI binaries.

## Remediation applied
- Replaced the Python fallback in `_center` with a direct partial-pivot Gaussian elimination routine mirroring the C++ backend. This removes the dependency on `numpy.linalg.lstsq`'s version-specific behavior and brings both backends into numerical alignment for degenerate or ill-conditioned conics.

## Recommendations for future CI hardening
1. Extend the matrix to exercise multiple NumPy/SciPy baselines (e.g., lowest supported vs. latest) on at least one Python version so regressions from upstream LAPACK updates surface early.
2. Cache artifact versions (NumPy/SciPy wheel hashes) in CI logs for quicker comparison when divergences appear.
3. Add an opt-in workflow job that reruns the numerical stress suite with the Python backend only, using the same fallback solver, to ensure feature parity without relying on the C++ path to mask issues.
4. Document any known-bad NumPy/SciPy releases in `docs/` and gate them via dependency constraints if future regressions are confirmed.

## Next steps
- If additional divergence appears, bisect NumPy/SciPy releases starting from 2.2.6/1.15.3 to pinpoint offending wheels.
- Consider adding randomized but fixed-seed stress cases to detect subtle solver drift between releases.
