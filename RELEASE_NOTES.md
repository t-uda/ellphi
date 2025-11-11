## 0.1.1 - 2025-11-09

### Added

- Full n-dimensional support for tangency solving across Python and C++ backends, including new conic
packing helpers, EllipseCloud dimension tracking, and expanded differentiable gradient coverage with
high-dimensional regression tests.
- A dedicated 3-D ellipsoid demo notebook plus documentation of the ndim extension review.

### Fixed

- The C++ backend now validates coefficient/point buffer lengths and clamps negative quadratic
evaluations before taking square roots, while the Python wrapper rejects malformed coefficient arrays
early.

### Changed

- Dropped support for Python 3.9. Python 3.10 or newer is now required.

### Tooling

- AGENTS.md documents the exact CI command checklist and black[jupyter] is added so notebooks are
formatted consistently.
