#!/bin/bash
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Configuration for cibuildwheel
export CIBW_ARCHS_LINUX="${CIBW_ARCHS_LINUX:-x86_64 aarch64}"
export CIBW_PLATFORM="linux"
export CIBW_MANYLINUX_X86_64_IMAGE="manylinux_2_28"
export CIBW_MANYLINUX_AARCH64_IMAGE="manylinux_2_28"

export CIBW_CACHE_PATH="${CIBW_CACHE_PATH:-$ROOT/build/cibuildwheel}"
export CIBW_BEFORE_BUILD_LINUX="if command -v yum >/dev/null 2>&1; then yum install -y openblas-devel pkgconf-pkg-config; elif command -v apk >/dev/null 2>&1; then apk add --no-cache openblas-dev pkgconf; fi"
export CIBW_ENVIRONMENT_LINUX="ELLPHI_USE_LAPACK=1 ELLPHI_LAPACK_LINK_ARGS='-lopenblas'"

export CIBW_TEST_COMMAND="python -c 'import ellphi; print(\"cpp backend:\", ellphi.has_cpp_backend())'"

# Output directory for wheels (default: wheelhouse)
OUTPUT_DIR="${1:-wheelhouse}"

# Run cibuildwheel
python3 -m cibuildwheel --output-dir "$OUTPUT_DIR"
