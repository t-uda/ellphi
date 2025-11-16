#!/bin/sh

set -e

mkdir -p build/benchmarks

# 1) Extreme vs non-extreme (baseline: default hybrid combos, dims 2/3)
poetry run python scripts/hybrid_tuning.py \
  --samples-per-dim 5000 --dims 2 --extreme-fraction 1.00 \
  --plot-dir build/benchmarks --plot-prefix extreme- \
  --output build/benchmarks/hybrid_tuning_summary_extreme.json \
  --warmup 1 \
  > build/benchmarks/hybrid_tuning_summary_extreme.md

poetry run python scripts/hybrid_tuning.py \
  --samples-per-dim 5000 --dims 2 --extreme-fraction 0.00 \
  --plot-dir build/benchmarks --plot-prefix nonextreme- \
  --output build/benchmarks/hybrid_tuning_summary_nonextreme.json \
  --warmup 1 \
  > build/benchmarks/hybrid_tuning_summary_nonextreme.md

poetry run python scripts/hybrid_tuning.py \
  --samples-per-dim 5000 --dims 3 --extreme-fraction 1.00 \
  --plot-dir build/benchmarks --plot-prefix extreme3d- \
  --output build/benchmarks/hybrid_tuning_summary_extreme3d.json \
  --warmup 1 \
  > build/benchmarks/hybrid_tuning_summary_extreme3d.md

poetry run python scripts/hybrid_tuning.py \
  --samples-per-dim 5000 --dims 3 --extreme-fraction 0.00 \
  --plot-dir build/benchmarks --plot-prefix nonextreme3d- \
  --output build/benchmarks/hybrid_tuning_summary_nonextreme3d.json \
  --warmup 1 \
  > build/benchmarks/hybrid_tuning_summary_nonextreme3d.md

# 2) Custom hybrid combos (override iteration counts)
poetry run python scripts/hybrid_tuning.py \
  --samples-per-dim 5000 --dims 2 3 4 5 \
  --extreme-fraction 0.45 \
  --hybrid-combos "8x3,16x1,16x3,28x1,28x3,64x1,64x3" \
  --plot-dir build/benchmarks --plot-prefix lowdim- \
  --output build/benchmarks/hybrid_tuning_summary_lowdim.json \
  --warmup 1 \
  > build/benchmarks/hybrid_tuning_summary_lowdim.md

# 3) Fixed-case reuse: generate once, then rerun different backends/combos on identical data
poetry run python scripts/hybrid_tuning.py \
  --samples-per-dim 5000 --dims 10 20 30 \
  --extreme-fraction 0.45 \
  --cases-output build/benchmarks/cases_fixed.json \
  --plot-dir build/benchmarks --plot-prefix highdim- \
  --output build/benchmarks/hybrid_tuning_summary_highdim.json \
  --warmup 1 \
  > build/benchmarks/hybrid_tuning_summary_highdim.md

# Reuse the same cases with a different hybrid combo set
poetry run python scripts/hybrid_tuning.py \
  --cases-input build/benchmarks/cases_fixed.json \
  --hybrid-combos "8x3,16x1,16x3,28x1,28x3,64x1,64x3" \
  --plot-dir build/benchmarks --plot-prefix highdim-alt- \
  --output build/benchmarks/hybrid_tuning_summary_highdim_alt.json \
  --warmup 1 \
  > build/benchmarks/hybrid_tuning_summary_highdim_alt.md

