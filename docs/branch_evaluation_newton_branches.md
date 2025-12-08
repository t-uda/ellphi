# Branch evaluation & PR comparison: Newton/hybrid solver directions

## Scope & methodology
- Focus on Newton/(algsig+)Newton behaviour per `docs/newton-solver-alignment-report.md` と `docs/newton_algsig_performance_report.md`.
- Bench: `scripts/hybrid_tuning.py --samples-per-dim 5000 --dims 2 3 --extreme-fraction 0.0 --methods newton algsig+newton --failsafe-options false true --backends python cpp` on `feat/algsig-newton` (PR #54), `fix/numerical-stability-issue` (#55), `codex/investigate-ci-python-library-versions` and variants #56/#57/#58/#59, and `fix/newton-diverge-antigravity-gemini-3` (現HEAD). Outputs: `/tmp/ellphi-bench/*.txt`/`.json`.
- Criterion: keep Python/C++ parity in Newton/`algsig+newton`; avoid Brent as the primary strategy; remove version-dependent fallbacks.

## Normal lowdim results (Python backend, failures over 10k samples)
- `algsig+newton` (failsafe=false): **590 failures** on all branches except `fix/numerical-stability-issue` (0 failures only because it routes to Brent, not Newton).  
- `algsig+newton` (failsafe=true): 0 failures across branches.  
- `newton` (failsafe=false): 37 failures across branches.  
- C++ backend: 0 failures for all methods/branches.

## Consolidated comparison (PR/branch)
| Branch / PR | CI status | Newton strategy & fallback | Key changes / review notes | Normal lowdim (Py) | Trade-offs |
|-------------|-----------|----------------------------|----------------------------|--------------------|------------|
| `feat/algsig-newton` (#54) | completed / failure | Newton path with `lstsq` fallback | Adds `algsig+newton`, xtol/rtol guards; `lstsq` is version-sensitive | `algsig+newton` fails 5.9% (failsafe=false) | Needs parity fix; fallback differs from C++ |
| `fix/numerical-stability-issue` (#55) | completed / success | **Brent substitution** for `algsig+newton` | [P1] Newton path bypassed; hides issue, no parity | 0 failures only because Newton skipped | Not acceptable for Newton-focused work |
| `codex/investigate-ci-python-library-versions` (#56) | completed / failure | Pivoted Gaussian fallback | Adds `_gaussian_elimination`; version doc | `algsig+newton` fails 5.9% (failsafe=false) | Better linear-solve parity; algsig still diverges |
| `codex/investigate-ci-python-library-versions-atlq32` (#57) | completed / failure | Pivoted Gaussian fallback | Same as #56 with doc tweaks | `algsig+newton` fails 5.9% (failsafe=false) | Same risk as #56 |
| `codex/investigate-ci-python-library-versions-7497h8` (#58) | completed / failure | Pivoted Gaussian fallback + tests/stubs | Adds stub/test coverage for fallback | `algsig+newton` fails 5.9% (failsafe=false) | Coverage improved; divergence remains |
| `codex/investigate-ci-python-library-versions-8xs2mp` (#59) | completed / success | Pivoted Gaussian fallback + version doc | Stability doc | `algsig+newton` fails 5.9% (failsafe=false) | Docs updated; divergence remains |
| `fix/newton-diverge-antigravity-gemini-3` (HEAD) | n/a | LU (`scipy.linalg.solve`) fallback | [P2] NaNs on singular pencils; no pivoted fallback | `algsig+newton` fails 5.9% (failsafe=false) | Closer than `lstsq` but still diverges |

## Latest review findings
- [P1] `fix/numerical-stability-issue` routes `algsig+newton` to Brent, so Newton is not exercised and parity is lost.
- [P2] `fix/newton-diverge-antigravity-gemini-3` falls back to `scipy.linalg.solve` and returns NaN on singular/ill-conditioned cases; needs pivoted Gaussian fallback like `codex/*` to match C++ and avoid the 5.9% divergence.

## Recommendations (Brent fallbackを主戦略にしない)
- Keep the Newton path intact (avoid Brent substitution from #55).
- Adopt the pivoted Gaussian fallback (from `codex/*`) in current HEAD to match C++ instead of `lstsq`/plain `solve`.
- Align Python `algsig_newton` step acceptance/guards with C++ to remove the 5.9% failures when failsafe=false; use `--find-divergent-case` to capture repro inputs.
- After solver fixes, rerun the same normal lowdim bench to confirm `algsig+newton`(failsafe=false) hits 0 failures without Brent.

## Test/CI gaps
- Add deterministic regression for divergent cases once captured to enforce Python/C++ parity for `algsig+newton` without Brent or failsafe.
- Update `newton-solver-alignment-report.md` after the linear solver and algsig parity fixes.***
