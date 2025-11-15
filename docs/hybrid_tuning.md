# Hybrid Solver Parameter Tuning

We introduced `scripts/hybrid_tuning.py` to benchmark
`ellphi.solver.tangency` across challenging ellipsoid configurations.
The script generates 30 test pairs per dimension for dims 2–5 with 45%
of the cases featuring axis-ratio extrema up to 10³.  Each case is run
through the single-stage methods (bisect, brentq, brenth, Newton) and
multiple hybrid iteration budgets, for every requested backend
(`python` and `cpp` when available).  Results are printed as a Markdown
table with aggregated and per-dimension runtime/accuracy metrics (so 2D
vs. n-D comparisons and backend differences are explicit) and the same
data is emitted to `--output` in JSON format.  Use `--table-format
plain` if you prefer the previous text-based layout.  Passing
`--plot-dir <path>` will additionally output log-log scatter plots of
median time vs. relative error for each backend; `--plot-prefix` keeps
multiple runs from overwriting files.  `--hybrid-combos` overrides the
iteration pairs (e.g., `--hybrid-combos "8x3,16x5"`), and
`--cases-input/--cases-output` let you reuse or persist the exact case
set.  All
numbers below come from running:

```bash
poetry run python scripts/hybrid_tuning.py \
  --samples-per-dim 30 --dims 2 3 4 5 --extreme-fraction 0.45 \
  --output docs/hybrid_tuning_summary.json
```

Errors are reported as the relative residual of the tangency conditions,
namely the maximum of (i) the difference between the two quadratic forms
evaluated at the reported contact point and (ii) the mismatch between
those quadratic values and the reported scaling factor, normalised by
`t^2`.

## Key Findings

* `brentq+newton` with 28 Brent iterations followed by 6 Newton steps
  delivered zero failures across 120 cases while keeping mean absolute
  μ error below 5×10⁻⁹.
* The historical 2D tuning (8 Brent / 3 Newton iterations) remains
  optimal for planar ellipses, so the library keeps that shortcut and
  only ups the budget for n>2 configurations.
* The tuned hybrid (`28×6`) halves the mean runtime compared to
  pure bisection (1.09 ms vs. 2.32 ms) while retaining the robustness of
the bracketing stage.
* Plain Newton with a naïve `x0=0.5` frequently diverges (28/120
  failures), underscoring the need for the hybrid fallback, especially
  when handling extreme aspect ratios.
* Lower iteration budgets such as `12×4` or `16×4` are attractive in 2D
  but failed in ≥10% of the stress cases; keeping the bracket stage at
  28 iterations ensures convergence in higher dimensions without a large
  runtime penalty.

See `docs/hybrid_tuning_summary.json` for the full aggregated metrics per
method as emitted by the script.
