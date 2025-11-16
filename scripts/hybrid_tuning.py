#!/usr/bin/env python3
"""Benchmark tangency backends/methods to tune the hybrid solver.

The script generates a mix of well-conditioned and highly eccentric
ellipsoids, calls :func:`ellphi.solver.tangency` with different
backends/methods, and reports aggregated + per-dimension timing/accuracy
statistics (markdown table by default).  The output guides the selection
of iteration budgets that generalise beyond 2D while remaining faster
than single-stage root finders.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import itertools
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ellphi.geometry import coef_from_cov
from ellphi.solver import tangency, has_cpp_backend, quad_eval


@dataclass
class Case:
    dim: int
    p: np.ndarray
    q: np.ndarray


def _resolve_backends(requested: Sequence[str]) -> list[str]:
    resolved: list[str] = []
    seen: set[str] = set()
    cpp_available = has_cpp_backend()
    for backend in requested or ["python"]:
        name = backend.lower()
        if name not in {"python", "cpp"}:
            raise ValueError(
                f"Unknown backend '{backend}'. Expected 'python' or 'cpp'."
            )
        if name == "cpp" and not cpp_available:
            print(
                "Skipping backend 'cpp' because the C++ backend is not available.",
                file=sys.stderr,
            )
            continue
        if name in seen:
            continue
        resolved.append(name)
        seen.add(name)
    if not resolved:
        raise RuntimeError("No usable backends selected for benchmarking.")
    return resolved


def _random_rotation(dim: int, rng: np.random.Generator) -> np.ndarray:
    mat = rng.standard_normal((dim, dim))
    q, _ = np.linalg.qr(mat)
    return q


def _covariance(dim: int, rng: np.random.Generator, extreme: bool) -> np.ndarray:
    if extreme:
        axes = rng.uniform(0.01, 100.0, size=dim)
    else:
        axes = rng.uniform(0.5, 2.0, size=dim)
    rot = _random_rotation(dim, rng)
    cov = rot @ np.diag(axes**2) @ rot.T
    jitter = 1e-9 * np.eye(dim, dtype=float)
    return cov + jitter


def _sample_case(dim: int, rng: np.random.Generator, extreme: bool) -> Case:
    attempts = 0
    while True:
        attempts += 1
        means = rng.uniform(-25.0, 25.0, size=(2, dim))
        covs = np.stack([_covariance(dim, rng, extreme) for _ in range(2)], axis=0)
        try:
            coefs = coef_from_cov(means, covs)
            return Case(dim=dim, p=coefs[0], q=coefs[1])
        except np.linalg.LinAlgError:
            if attempts >= 5:
                raise


def _build_cases(
    dims: Sequence[int],
    samples_per_dim: int,
    extreme_fraction: float,
    rng: np.random.Generator,
) -> list[Case]:
    cases: list[Case] = []
    for dim in dims:
        n_extreme = max(1, int(round(samples_per_dim * extreme_fraction)))
        for _ in range(samples_per_dim):
            extreme = len(cases) % samples_per_dim < n_extreme
            cases.append(_sample_case(dim, rng, extreme=extreme))
    return cases


def _save_cases(cases: Sequence[Case], path: Path) -> None:
    payload = [
        {"dim": case.dim, "p": case.p.tolist(), "q": case.q.tolist()} for case in cases
    ]
    path.write_text(json.dumps(payload, indent=2))


def _load_cases(path: Path) -> list[Case]:
    payload = json.loads(path.read_text())
    cases: list[Case] = []
    for item in payload:
        dim = int(item["dim"])
        p = np.asarray(item["p"], dtype=float)
        q = np.asarray(item["q"], dtype=float)
        cases.append(Case(dim=dim, p=p, q=q))
    return cases


def _relative_tangency_residual(p: np.ndarray, q: np.ndarray, result) -> float:
    value_p = quad_eval(p, result.point)
    value_q = quad_eval(q, result.point)
    t_sq = float(result.t) ** 2
    numer = abs(value_p - value_q) + abs(value_p - t_sq) + abs(value_q - t_sq)
    denom = abs(t_sq)
    return numer / denom if denom != 0.0 else numer


def _stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean": math.nan, "median": math.nan, "p99": math.nan, "samples": 0}
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "p99": float(np.percentile(values, 99.0)),
        "samples": len(values),
    }


def _evaluate_case(
    case: Case,
    combos: Iterable[tuple[int, int]],
    benchmark_methods: Iterable[str],
    backends: Sequence[str],
) -> dict[str, dict[str, float]]:
    method_specs: list[tuple[str, dict[str, object]]] = []
    for method in benchmark_methods:
        kwargs: dict[str, object] = {"method": method}
        if method == "newton":
            kwargs["x0"] = 0.5
        method_specs.append((method, kwargs))

    hybrid_specs: list[tuple[str, dict[str, object]]] = [
        (
            f"hybrid_{b_iter}x{n_iter}",
            {
                "method": "brentq+newton",
                "hybrid_bracket_maxiter": b_iter,
                "hybrid_newton_maxiter": n_iter,
            },
        )
        for b_iter, n_iter in combos
    ]

    all_specs = [*method_specs, *hybrid_specs]

    per_case: dict[str, dict[str, float]] = {}
    for backend in backends:

        def label_to_key(label: str) -> str:
            return f"{backend}:{label}"

        for label, base_kwargs in all_specs:
            key = label_to_key(label)
            kwargs = dict(base_kwargs)
            kwargs["backend"] = backend
            start = time.perf_counter()
            try:
                result = tangency(case.p, case.q, **kwargs)
                elapsed = time.perf_counter() - start
                error = _relative_tangency_residual(case.p, case.q, result)
                per_case[key] = {"time": elapsed, "error": error}
            except Exception:
                per_case[key] = {}

    return per_case


def _summarize(
    cases: Sequence[Case],
    combos: Sequence[tuple[int, int]],
    benchmark_methods: Sequence[str],
    backends: Sequence[str],
) -> tuple[dict[str, dict[str, dict[str, dict[str, float]]]], list[int]]:
    metrics: dict[str, list[float]] = defaultdict(list)
    errors: dict[str, list[float]] = defaultdict(list)
    failures: Counter[str] = Counter()

    metrics_by_dim: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    errors_by_dim: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    failures_by_dim: dict[int, Counter[str]] = defaultdict(Counter)

    dims_in_suite = sorted({case.dim for case in cases})
    all_keys: list[str] = []

    for case in cases:
        case_stats = _evaluate_case(case, combos, benchmark_methods, backends)
        dim = case.dim
        for key, values in case_stats.items():
            if key not in metrics:
                all_keys.append(key)
            if not values:
                failures[key] += 1
                failures_by_dim[dim][key] += 1
                continue
            metrics[key].append(values["time"])
            errors[key].append(values["error"])
            metrics_by_dim[dim][key].append(values["time"])
            errors_by_dim[dim][key].append(values["error"])

    summary: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for key in sorted(all_keys):
        per_dim: dict[str, dict[str, dict[str, float]]] = {}
        for dim in dims_in_suite:
            metrics_dim = metrics_by_dim.get(dim, {})
            errors_dim = errors_by_dim.get(dim, {})
            per_dim[str(dim)] = {
                "time": _stats(metrics_dim.get(key, [])),
                "error": _stats(errors_dim.get(key, [])),
                "failures": {"count": failures_by_dim.get(dim, Counter()).get(key, 0)},
            }
        summary[key] = {
            "overall": {
                "time": _stats(metrics[key]),
                "error": _stats(errors[key]),
                "failures": {"count": failures.get(key, 0)},
            },
            "per_dim": per_dim,
        }
    return summary, dims_in_suite


def _build_table_rows(
    summary: dict[str, dict[str, dict[str, dict[str, float]]]],
    dims: Sequence[int],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def _fmt_stat(value: float, scale: float = 1.0) -> str:
        if math.isnan(value):
            return "nan"
        return f"{value * scale:.4e}"

    def _make_row(
        backend: str,
        method: str,
        scope: str,
        dim_label: str,
        stats: dict[str, dict[str, float]],
    ) -> dict[str, str]:
        time_stats = stats["time"]
        err_stats = stats["error"]
        failures = stats["failures"]["count"]
        return {
            "Backend": backend,
            "Method": method,
            "Scope": scope,
            "Dim": dim_label,
            "Time Mean (ms)": _fmt_stat(time_stats["mean"], 1000.0),
            "Time Median (ms)": _fmt_stat(time_stats["median"], 1000.0),
            "Time P99 (ms)": _fmt_stat(time_stats["p99"], 1000.0),
            "Error Mean": _fmt_stat(err_stats["mean"], 1.0),
            "Error Median": _fmt_stat(err_stats["median"], 1.0),
            "Error P99": _fmt_stat(err_stats["p99"], 1.0),
            "Failures": str(failures),
        }

    for key in sorted(summary):
        backend, method = key.split(":", 1)
        overall = summary[key]["overall"]
        rows.append(_make_row(backend, method, "overall", "-", overall))
        for dim in dims:
            dim_stats = summary[key]["per_dim"].get(str(dim))
            if dim_stats is None:
                continue
            rows.append(_make_row(backend, method, "dim", str(dim), dim_stats))
    return rows


def _print_markdown_table(rows: Sequence[dict[str, str]]) -> None:
    if not rows:
        print("No data to display")
        return

    headers = list(rows[0].keys())
    header_line = " | ".join(headers)
    separator_line = " | ".join(["---" for _ in headers])
    print(f"| {header_line} |")
    print(f"| {separator_line} |")
    for row in rows:
        line = " | ".join(row[h] for h in headers)
        print(f"| {line} |")


def _plot_backend_scatter(
    summary: dict[str, dict[str, dict[str, dict[str, float]]]],
    backends: Sequence[str],
    output_dir: Path,
    prefix: str = "",
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"Skipping plot generation ({exc})", file=sys.stderr)
        return

    methods = sorted({key.split(":", 1)[1] for key in summary})
    markers = [
        "o",
        "s",
        "^",
        "D",
        "v",
        ">",
        "<",
        "p",
        "P",
        "X",
        "*",
        "h",
        "d",
        "1",
        "2",
    ]
    colors = plt.cm.tab20.colors
    combos = list(zip(markers, colors))
    style_map: dict[str, tuple[str, tuple[float, float, float, float]]] = {}
    for idx, method in enumerate(methods):
        marker, color = combos[idx % len(combos)]
        style_map[method] = (marker, color)

    for backend in backends:
        method_points: dict[str, tuple[list[float], list[float]]] = {}
        for method in methods:
            key = f"{backend}:{method}"
            entry = summary.get(key)
            if not entry:
                continue
            time_stats = entry["overall"]["time"]
            err_stats = entry["overall"]["error"]
            time_value = time_stats.get("median")
            err_value = err_stats.get("mean")
            if (
                time_value is None
                or err_value is None
                or time_value <= 0.0
                or err_value <= 0.0
            ):
                continue
            xs_list, ys_list = method_points.setdefault(method, ([], []))
            xs_list.append(time_value * 1e3)
            ys_list.append(err_value)
        if not method_points:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        for method, (xs_vals, ys_vals) in method_points.items():
            marker, color = style_map[method]
            ax.scatter(
                xs_vals,
                ys_vals,
                marker=marker,
                color=color,
                edgecolor="black",
                linewidth=0.25,
                s=70,
                label=method,
            )
        handles, handle_labels = ax.get_legend_handles_labels()
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Median time per tangency (ms)")
        ax.set_ylabel("Relative tangency residual")
        ax.set_title(f"Hybrid benchmark: time vs error ({backend})")
        ax.grid(True, which="both", ls="--", alpha=0.4)
        ax.legend(handles, handle_labels, loc="best", fontsize=7, ncol=2)
        fig.tight_layout()
        outfile = output_dir / f"{prefix}hybrid_time_vs_error_{backend}.png"
        fig.savefig(outfile, dpi=200)
        plt.close(fig)
        print(f"Wrote {outfile}", file=sys.stderr)


def _default_combos() -> list[tuple[int, int]]:
    return [
        (8, 3),
        (12, 4),
        (16, 4),
        (16, 5),
        (20, 5),
        (24, 5),
        (24, 6),
        (28, 6),
        (32, 6),
    ]


def _parse_combos(arg: str | None) -> list[tuple[int, int]]:
    if not arg:
        return _default_combos()
    combos: list[tuple[int, int]] = []
    for chunk in arg.split(","):
        text = chunk.strip()
        if not text:
            continue
        if "x" not in text:
            raise ValueError(f"Invalid combo '{text}', expected form <brent>x<newton>")
        a, b = text.split("x", 1)
        combos.append((int(a), int(b)))
    if not combos:
        return _default_combos()
    return combos


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-dim", type=int, default=80)
    parser.add_argument("--dims", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--extreme-fraction", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["python", "cpp"],
        help="Backends to benchmark (python, cpp)",
    )
    parser.add_argument(
        "--table-format",
        choices=("markdown", "plain"),
        default="markdown",
        help="Format used for the on-screen summary table",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        help="Optional directory to write per-backend time/error scatter plots",
    )
    parser.add_argument(
        "--plot-prefix",
        type=str,
        default="",
        help="Optional prefix added to generated plot filenames",
    )
    parser.add_argument(
        "--cases-input",
        type=Path,
        help="Load pre-generated cases from JSON instead of sampling",
    )
    parser.add_argument(
        "--cases-output",
        type=Path,
        help="Write the generated cases to JSON for reuse",
    )
    parser.add_argument(
        "--hybrid-combos",
        type=str,
        help="Comma-separated list of hybrid iteration pairs, e.g. '8x3,16x5'",
    )
    parser.add_argument(
        "--warmup", type=int, default=0, help="Number of warmup evaluations to skip"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)
    backends = _resolve_backends(args.backends)
    combos = _parse_combos(args.hybrid_combos)
    benchmarks = ["bisect", "brentq", "brenth", "newton"]
    if args.cases_input:
        cases = _load_cases(args.cases_input)
    else:
        cases = _build_cases(
            args.dims, args.samples_per_dim, args.extreme_fraction, rng
        )
        if args.cases_output:
            _save_cases(cases, args.cases_output)
    if args.warmup > 0:
        cases = cases[args.warmup :]
    summary, dims = _summarize(cases, combos, benchmarks, backends)

    def _fmt(stat: dict[str, float]) -> str:
        return (
            f"mean={stat['mean']:.6f}s median={stat['median']:.6f}s "
            f"p99={stat['p99']:.6f}s"
        )

    def _fmt_err(stat: dict[str, float]) -> str:
        return (
            f"error(mean={stat['mean']:.3e}, "
            f"median={stat['median']:.3e}, p99={stat['p99']:.3e})"
        )

    if args.plot_dir:
        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        _plot_backend_scatter(summary, backends, plot_dir, prefix=args.plot_prefix)

    if args.table_format == "markdown":
        rows = _build_table_rows(summary, dims)
        _print_markdown_table(rows)
    else:
        for key in sorted(summary):
            backend, method = key.split(":", 1)
            overall = summary[key]["overall"]
            time_stats = overall["time"]
            err_stats = overall["error"]
            fail_count = overall["failures"]["count"]
            overall_line = (
                f"[{backend}] {method:<16} time({_fmt(time_stats)}) "
                f"{_fmt_err(err_stats)} failures={fail_count}"
            )
            print(overall_line)
            for dim in dims:
                dim_stats = summary[key]["per_dim"].get(str(dim))
                if dim_stats is None:
                    continue
                dim_time = dim_stats["time"]
                dim_err = dim_stats["error"]
                dim_fail = dim_stats["failures"]["count"]
                dim_line = (
                    f"  dim={dim:<5} time({_fmt(dim_time)}) "
                    f"{_fmt_err(dim_err)} failures={dim_fail}"
                )
                print(dim_line)

    if args.output:
        args.output.write_text(json.dumps(summary, indent=2))
        print(f"Wrote summary to {args.output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
