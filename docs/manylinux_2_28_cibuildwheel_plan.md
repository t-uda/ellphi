# manylinux_2_28 cibuildwheel plan for LAPACK-enabled wheels

## Goal
Provide Linux wheels that are as fast as possible by default. That means
publishing manylinux_2_28 wheels built with `ELLPHI_USE_LAPACK=1` and a bundled
OpenBLAS so `pip install ellphi` on Linux uses LAPACK without extra flags.

This plan assumes:
- Linux wheels target glibc >= 2.28 (manylinux_2_28).
- macOS/Windows can remain on the current default for now.
- The C++ backend is built by `scripts/build_tangency_cpp.py`.

## Key build knobs in this repo
The build script currently supports:
- `ELLPHI_USE_LAPACK=1` to enable LAPACK code paths at compile time.
- `ELLPHI_LAPACK_LINK_ARGS` to supply linker flags (e.g. from `pkg-config`).

These are compile-time switches, so the wheel must be built with LAPACK enabled.

## CI build outline (Linux wheel with OpenBLAS)
Use a release-only workflow (manual trigger or tags) so normal PR CI stays fast.
Limit the matrix to Python 3.10/3.11 and at most 2-3 architectures, then build
manylinux_2_28 wheels with OpenBLAS bundled.

Example CI step (conceptual):
```bash
# inside CI job
python -m pip install cibuildwheel

export CIBW_BUILD="cp310-* cp311-*"
export CIBW_ARCHS_LINUX="x86_64 aarch64"
export CIBW_SKIP="*-musllinux*"
export CIBW_MANYLINUX_X86_64_IMAGE="manylinux_2_28"
export CIBW_MANYLINUX_AARCH64_IMAGE="manylinux_2_28"

export CIBW_BEFORE_BUILD_LINUX="yum install -y openblas-devel pkgconf-pkg-config"
export CIBW_ENVIRONMENT_LINUX="ELLPHI_USE_LAPACK=1"

export CIBW_TEST_COMMAND="python - <<'PY'
import ellphi
print('cpp', ellphi.has_cpp_backend())
PY"

export CIBW_REPAIR_WHEEL_COMMAND_LINUX="auditwheel repair -w {dest_dir} {wheel}"

cibuildwheel --output-dir wheelhouse
```

If you need a third architecture, add it explicitly (for example, `ppc64le`)
instead of building every available target.

Note on linker flags:
- `ELLPHI_LAPACK_LINK_ARGS` must include the correct OpenBLAS link flags.
- Because `CIBW_ENVIRONMENT_LINUX` is static text, compute the flags in CI and
  pass them in the same shell that runs `cibuildwheel`.

Example pattern:
```bash
export ELLPHI_LAPACK_LINK_ARGS="$(pkg-config --libs openblas)"
export CIBW_ENVIRONMENT_LINUX="ELLPHI_USE_LAPACK=1 ELLPHI_LAPACK_LINK_ARGS=$ELLPHI_LAPACK_LINK_ARGS"
cibuildwheel --output-dir wheelhouse
```

## auditwheel and bundling
`auditwheel repair` will copy required shared libs into the wheel and update
the wheel tags. This is required because OpenBLAS is not a manylinux-whitelisted
system dependency.

Validate the output:
```bash
auditwheel show wheelhouse/*.whl
```
Ensure `libopenblas` (and any other dependencies) are shown as bundled.

## Licensing obligations (MIT + OpenBLAS + transitive libs)
MIT is compatible with OpenBLAS (BSD 3-clause). The key requirement is to
include the OpenBLAS license text in the wheel or sdist distribution and keep
copyright notices.

OpenBLAS often pulls in additional runtime libraries:
- `libgfortran` / `libquadmath` (GCC runtime, GPL with runtime exception)
- `libgomp` (if OpenMP is used)

These are generally redistributable, but their license texts must be included.

Recommended approach:
- Add a `LICENSES/` directory in the repo.
- During wheel build, collect license texts for every bundled library listed by
  `auditwheel show` and copy them into the wheel (or include them in sdist and
  ensure they land in the wheel).
- Add a short `NOTICE` file that references OpenBLAS and any other bundled libs.

Example checklist:
1) Run `auditwheel show dist/*.whl`.
2) For each bundled lib, record license and include `LICENSES/<name>.LICENSE`.
3) Ensure `LICENSES/` is part of `tool.poetry.include`.

## Developer workflow on macOS
macOS cannot run `auditwheel` directly. Use Docker and the manylinux image.

Two options:
1) Run `cibuildwheel` on macOS and let it use Docker internally.
2) Run a one-off build inside the container.

Example manual container workflow:
```bash
docker run --rm -it -v "$PWD:/project" -w /project quay.io/pypa/manylinux_2_28_x86_64 /bin/bash
# inside container:
yum install -y python3 python3-pip openblas-devel pkgconf-pkg-config
python3 -m pip install cibuildwheel
export ELLPHI_LAPACK_LINK_ARGS="$(pkg-config --libs openblas)"
export CIBW_ENVIRONMENT_LINUX="ELLPHI_USE_LAPACK=1 ELLPHI_LAPACK_LINK_ARGS=$ELLPHI_LAPACK_LINK_ARGS"
cibuildwheel --output-dir wheelhouse
auditwheel show wheelhouse/*.whl
```

## Validation checklist
- `pip install` the wheel in a clean Linux venv and run a small tangency call.
- Confirm `ellphi.has_cpp_backend()` is `True`.
- Compare a quick benchmark with and without LAPACK in a controlled environment.
- Run `ldd` on `_tangency_cpp_impl.so` to ensure all bundled libs resolve.

## Benchmark comparison (Eigen vs LAPACK vs internal)
Notes:
- Host: Ubuntu aarch64 (local dev machine)
- Eigen: 3.4.0 (`libeigen3-dev`)
- LAPACK: OpenBLAS 0.3.26 (`libopenblas-dev`)
- Build: deleted `_tangency_cpp_impl.so` then rebuilt for each mode via `poetry install`
- Run: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 poetry run python scripts/benchmark_dim_scale.py`
- joblib emitted a permissions warning and ran in serial mode

CPP results (avg_time_ms):
```
dim,Eigen,LAPACK,None
2,0.0245,0.0257,0.0235
3,0.0243,0.0262,0.0232
5,0.0262,0.0273,0.0243
8,0.0305,0.0307,0.0277
10,0.0338,0.0335,0.0301
15,0.0457,0.0479,0.0426
20,0.0560,0.0632,0.0583
30,0.0928,0.1081,0.1663
40,0.1570,0.1393,0.2003
50,0.2404,0.2283,0.3486
60,0.3323,0.3041,0.5255
80,0.5687,0.5350,1.0683
100,0.9809,0.8903,2.3383
```

## Known pitfalls and mitigations
- OpenBLAS threading can oversubscribe CPU. Consider documenting
  `OPENBLAS_NUM_THREADS=1` for reproducible benchmarks.
- manylinux_2_28 requires glibc >= 2.28. Older distros will not accept these
  wheels. If older compatibility is required, also build manylinux2014 wheels.
- For aarch64 on macOS, builds are slow without native Linux hardware. Prefer CI.

## Publishing to PyPI
1) Build wheels with `cibuildwheel` (Linux manylinux_2_28).
2) Build macOS/Windows wheels as usual (without LAPACK for now).
3) Upload all wheels and the sdist.
4) Verify the Linux wheel is picked by `pip` and contains OpenBLAS.
