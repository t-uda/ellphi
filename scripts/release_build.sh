#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Clean dist/ and build sdist + local wheel.
poetry build --clean

# Build manylinux wheels into dist/ for poetry publish.
scripts/manylinux_build.sh dist
