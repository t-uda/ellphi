# Eigen backend plan

## Decision
We will keep Eigen as the only optional accelerated backend. For moderate dimensions
(~100x100), Eigen provides enough speed without extra build and packaging
complexity. As a result, we do not maintain a special manylinux cibuildwheel
pipeline.

## Dependencies (C++)
- Optional: Eigen headers. Required only when building with `ELLPHI_USE_EIGEN=1`.
- Default builds do not require Eigen and use the internal C++ solver.

## Build knobs in this repo
The build script supports:
- `ELLPHI_USE_EIGEN=1` to enable Eigen code paths at compile time.
- `ELLPHI_EIGEN_INCLUDE` to set Eigen include paths (space-separated).

Notes:
- Eigen is header-only, so no runtime shared libraries are bundled.

## Build without Eigen
If Eigen headers are not available, build without enabling it:
1) Do not set `ELLPHI_USE_EIGEN`.
2) Run `uv sync` to build the C++ extension with the internal solver.
3) If `ELLPHI_USE_EIGEN=1` is set without headers, the build fails with a
   clear error asking for `ELLPHI_EIGEN_INCLUDE`.

## Local developer workflow
1) Install Eigen headers (examples):
   - Ubuntu: `apt-get install libeigen3-dev`
   - macOS (Homebrew): `brew install eigen`
2) If Eigen is not in a standard include location, set:
   - `ELLPHI_EIGEN_INCLUDE="/path/to/eigen3"`
3) Rebuild the extension:
   - `uv sync --reinstall-package ellphi`
   - If needed, delete `src/ellphi/_tangency_cpp_impl*.so` before reinstalling.
4) Quick sanity check:
   ```bash
   python - <<'PY'
   import ellphi; print(ellphi.has_cpp_backend())
   PY
   ```

## CI/release approach
- Use the normal wheel build process; no manylinux customization is required
  for Eigen because it is header-only.
- If Eigen-enabled wheels are desired, set `ELLPHI_USE_EIGEN=1` and
  `ELLPHI_EIGEN_INCLUDE` in the build environment.
- No `auditwheel` bundling is needed.

## Benchmark summary (Eigen vs internal)
Notes:
- Host: Ubuntu aarch64 (local dev machine)
- Eigen: 3.4.0 (`libeigen3-dev`)
- Build: deleted `_tangency_cpp_impl.so` then rebuilt for each mode via `uv sync --reinstall-package ellphi`
- Run: `OMP_NUM_THREADS=1 uv run python scripts/benchmark_dim_scale.py`
- joblib emitted a permissions warning and ran in serial mode

CPP results (avg_time_ms):
```
dim,Eigen,Internal
2,0.0245,0.0235
3,0.0243,0.0232
5,0.0262,0.0243
8,0.0305,0.0277
10,0.0338,0.0301
15,0.0457,0.0426
20,0.0560,0.0583
30,0.0928,0.1663
40,0.1570,0.2003
50,0.2404,0.3486
60,0.3323,0.5255
80,0.5687,1.0683
100,0.9809,2.3383
```

## Validation checklist
- Confirm `ellphi.has_cpp_backend()` returns `True` for Eigen builds.
- Confirm `ellphi.cpp_linalg_kind()` returns `eigen` when Eigen is enabled.
- Use `ellphi.build_info()` to inspect solver backend choices and C++ build details.
- Run a small tangency call or the benchmark script to confirm performance.
