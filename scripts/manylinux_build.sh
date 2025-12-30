#!/bin/bash
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Configuration for cibuildwheel
export CIBW_BUILD="cp312-manylinux_aarch64" # Limit to one version for testing
export CIBW_ARCHS_LINUX="aarch64"
export CIBW_PLATFORM="linux"
export CIBW_MANYLINUX_AARCH64_IMAGE="manylinux_2_28"

export CIBW_CACHE_PATH="${CIBW_CACHE_PATH:-$ROOT/build/cibuildwheel}"
export CIBW_BEFORE_BUILD_LINUX="yum install -y openblas-devel pkgconf-pkg-config"
export CIBW_ENVIRONMENT_LINUX="ELLPHI_USE_LAPACK=1 ELLPHI_LAPACK_LINK_ARGS='-lopenblas'"

export CIBW_TEST_COMMAND="python -c 'import ellphi; print(\"cpp backend:\", ellphi.has_cpp_backend())'"

# Output directory for wheels (default: wheelhouse)
OUTPUT_DIR="${1:-wheelhouse}"

# Run cibuildwheel
python3 -m cibuildwheel --output-dir "$OUTPUT_DIR"
