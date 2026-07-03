#!/bin/sh

set -e

mkdir -p build/benchmarks

# Common settings
SAMPLES=5000
SMALL_SAMPLES=500
HYBRID_COMBOS="28x3"
METHODS="brentq brenth algsig+newton newton"

# 1) Normal Low-Dim (2D, 3D) - The "Happy Path"
# Expect hybrid_8x3 to shine here.
uv run python scripts/hybrid_tuning.py \
  --samples-per-dim $SAMPLES --dims 2 3 \
  --extreme-fraction 0.0 \
  --methods $METHODS \
  --failsafe-options true false \
  --hybrid-combos "$HYBRID_COMBOS" \
  --plot-dir build/benchmarks --plot-prefix normal_lowdim- \
  --plot-type both \
  --output build/benchmarks/summary_normal_lowdim.json \
  --warmup 1 \
  > build/benchmarks/summary_normal_lowdim.md

# 2) Extreme Low-Dim (2D, 3D) - Stress Test
# Compare hybrid_8x3 failure rate vs hybrid_28x3/brent.
uv run python scripts/hybrid_tuning.py \
  --samples-per-dim $SAMPLES --dims 2 3 \
  --extreme-fraction 1.0 \
  --hybrid-combos "$HYBRID_COMBOS" \
  --plot-dir build/benchmarks --plot-prefix extreme_lowdim- \
  --plot-type both \
  --output build/benchmarks/summary_extreme_lowdim.json \
  --warmup 1 \
  > build/benchmarks/summary_extreme_lowdim.md

# 3) Normal High-Dim (10D, 20D) - Scaling Check
# Check if hybrid_8x3 scales to high dims in benign geometry.
uv run python scripts/hybrid_tuning.py \
  --samples-per-dim $SAMPLES --dims 10 20 \
  --extreme-fraction 0.0 \
  --hybrid-combos "$HYBRID_COMBOS" \
  --plot-dir build/benchmarks --plot-prefix normal_highdim- \
  --plot-type both \
  --output build/benchmarks/summary_normal_highdim.json \
  --warmup 1 \
  > build/benchmarks/summary_normal_highdim.md

# 4) Extreme High-Dim (10D, 20D) - The "Worst Case"
# Expect hybrid_8x3 to fail. Evaluate hybrid_28x3 vs brent.
uv run python scripts/hybrid_tuning.py \
  --samples-per-dim $SAMPLES --dims 10 20 \
  --extreme-fraction 1.0 \
  --hybrid-combos "$HYBRID_COMBOS" \
  --plot-dir build/benchmarks --plot-prefix extreme_highdim- \
  --plot-type both \
  --output build/benchmarks/summary_extreme_highdim.json \
  --warmup 1 \
  > build/benchmarks/summary_extreme_highdim.md

# 5) Multi-Dim (20D to 50D) - Seeking the "Strategy Shift"
# Check if hybrid still works for much higher dimention.
uv run python scripts/hybrid_tuning.py \
  --samples-per-dim $SMALL_SAMPLES --dims `seq 20 3 50` \
  --extreme-fraction 0.0 \
  --methods $METHODS \
  --hybrid-combos "$HYBRID_COMBOS" \
  --plot-dir build/benchmarks --plot-prefix multidim- \
  --plot-type both \
  --output build/benchmarks/summary_multidim.json \
  --warmup 1 \
  > build/benchmarks/summary_multidim.md

# 2) Extreme Low-Dim (2D, 3D) - Stress Test
# Compare hybrid_8x3 failure rate vs hybrid_28x3/brent.
uv run python scripts/hybrid_tuning.py \
  --samples-per-dim $SAMPLES --dims 2 3 \
  --extreme-fraction 1.0 \
  --methods $METHODS \
  --failsafe-options true false \
  --hybrid-combos "$HYBRID_COMBOS" \
  --plot-dir build/benchmarks --plot-prefix extreme_lowdim- \
  --plot-type both \
  --output build/benchmarks/summary_extreme_lowdim.json \
  --warmup 1 \
  > build/benchmarks/summary_extreme_lowdim.md

# 3) Normal High-Dim (10D, 20D) - Scaling Check
# Check if hybrid_8x3 scales to high dims in benign geometry.
uv run python scripts/hybrid_tuning.py \
  --samples-per-dim $SAMPLES --dims 10 20 \
  --extreme-fraction 0.0 \
  --methods $METHODS \
  --failsafe-options true false \
  --hybrid-combos "$HYBRID_COMBOS" \
  --plot-dir build/benchmarks --plot-prefix normal_highdim- \
  --plot-type both \
  --output build/benchmarks/summary_normal_highdim.json \
  --warmup 1 \
  > build/benchmarks/summary_normal_highdim.md

# 4) Extreme High-Dim (10D, 20D) - The "Worst Case"
# Expect hybrid_8x3 to fail. Evaluate hybrid_28x3 vs brent.
uv run python scripts/hybrid_tuning.py \
  --samples-per-dim $SAMPLES --dims 10 20 \
  --extreme-fraction 1.0 \
  --methods $METHODS \
  --hybrid-combos "$HYBRID_COMBOS" \
  --plot-dir build/benchmarks --plot-prefix extreme_highdim- \
  --plot-type both \
  --output build/benchmarks/summary_extreme_highdim.json \
  --warmup 1 \
  > build/benchmarks/summary_extreme_highdim.md

# 5) Multi-Dim (20D to 50D) - Seeking the "Strategy Shift"
# Check if hybrid still works for much higher dimention.
uv run python scripts/hybrid_tuning.py \
  --samples-per-dim $SMALL_SAMPLES --dims `seq 20 3 50` \
  --extreme-fraction 0.0 \
  --methods $METHODS \
  --failsafe-options true false \
  --hybrid-combos "$HYBRID_COMBOS" \
  --plot-dir build/benchmarks --plot-prefix multidim- \
  --plot-type both \
  --output build/benchmarks/summary_multidim.json \
  --warmup 1 \
  > build/benchmarks/summary_multidim.md
