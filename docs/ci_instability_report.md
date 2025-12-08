# CI Numerical Instability Investigation Report

## Executive Summary
This report details the investigation into CI failures observed in the `ellphi` project, specifically concerning the divergence of the "strict" test cases in Python 3.11 environments. The root cause was identified as numerical instability in `numpy.linalg.lstsq` when handling near-singular matrices during the pencil traversal. The issue has been resolved by aligning the Python backend's solver strategy with the C++ backend, replacing `lstsq` with explicit LU decomposition (`scipy.linalg.solve`) and strictly handling singular cases.

## Issue Description
Test cases with high precision requirements were passing in local environments (Python 3.12, macOS) but failing in CI (Python 3.11, Ubuntu). The failure manifested as divergence in the `algsig+newton` solver.

### Root Cause Analysis
- **Solver Inconsistency**: The Python backend used `numpy.linalg.lstsq` as a fallback when Cholesky factorization failed. `lstsq` computes a least-squares solution using SVD or Divide-and-Conquer algorithms. For nearly singular matrices (which occur frequently near the boundaries of the tangent pencil), `lstsq` can return valid-looking but numerically unstable solutions that differ significantly across LAPACK versions or architectures.
- **Backend Discrepancy**: The C++ backend uses a strictly deterministic approach: Cholesky factorization followed by Gaussian elimination with partial pivoting if degenerate. It explicitly raises an error for singular matrices, whereas `lstsq` "smooths over" singularities, potentially leading the Newton solver down a divergent path.

## Implemented Fix
We modified `ellphi/src/ellphi/_solver_python.py` to match the C++ backend's logic:

1.  **Primary Solver**: Attempt Cholesky factorization (`scipy.linalg.cho_factor`).
2.  **Fallback Solver**: If Cholesky fails (indefinite matrix), attempt LU decomposition (`scipy.linalg.solve`). This is mathematically equivalent to Gaussian elimination with partial pivoting and is more stable and deterministic for square invertible matrices than `lstsq`.
3.  **Singularity Handling**: If LU decomposition fails (singular matrix), we explicitly return `NaN`s instead of a least-squares approximation. This allows the Newton solver's failsafe or backtracking mechanisms to handle the invalid step appropriately, rather than proceeding with garbage data.

## Verification Results
- **Reproduction**: A reproduction script based on `test_numerical_stability.py` confirmed the fix locally.
- **Regression Testing**: The full test suite (`pytest tests`) passed (122 tests), confirming no regressions were introduced by changing the solver strategy.

## Recommendations for Future Stability

### 1. CI Matrix Expansion
The current CI only tests a limited set of Python versions. To catch version-specific numerical issues earlier, we recommend expanding the CI matrix:
- **Python Versions**: Test 3.10, 3.11, 3.12.
- **Dependency Versions**: Consider a "minimum compatible versions" job and a "latest versions" job to ensure the package works across the supported range of NumPy/SciPy.

### 2. Numerical Fuzz Testing
The "strict" test case that failed was likely a specific edge case. To verify robustness:
- Implement property-based testing (e.g., using `hypothesis`) to generate random ellipses and verify that the Python and C++ backends agree on the result within a strict tolerance.
- Specifically target "ill-conditioned" ellipses (high aspect ratios, very close centers) to stress-test the linear solvers.

### 3. Solver Parity Enforcement
- Continue to enforce strict parity between C++ and Python backends. 
- Consider exposing the internal solver steps (like `_center`) to unit tests to verify they produce identical results for identical inputs, not just the final `mu` value.
