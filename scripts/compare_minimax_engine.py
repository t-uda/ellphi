"""Compare the private minimax engine with its ellcech source revision.

This is an opt-in fidelity check, not part of the default pytest run. Generate
the shared inputs and the two engine records in separate processes, then
compare them::

    uv run --no-sync python scripts/compare_minimax_engine.py \
        inputs --output /tmp/minimax-inputs.json
    /path/to/ellcech/.venv/bin/python scripts/compare_minimax_engine.py \
        generate --engine legacy --revision 82d13e3 \
        --inputs /tmp/minimax-inputs.json --output /tmp/legacy.json
    /path/to/ellcech/.venv/bin/python scripts/compare_minimax_engine.py \
        generate --engine candidate --revision HEAD \
        --source-root . --inputs /tmp/minimax-inputs.json \
        --output /tmp/candidate.json
    uv run --no-sync python scripts/compare_minimax_engine.py \
        compare /tmp/legacy.json /tmp/candidate.json

Only cases for which the legacy engine reports ``converged=True`` are fidelity
requirements. ``METHOD_TOLERANCES`` are empirical comparison thresholds, not
solver settings; each method has a rationale recorded in ``METHOD_REASONS``.
For every method, records with ``alpha_legacy <= 1`` use the strict empirical
threshold, while records with larger alpha use a scale-aware threshold based on
``max(1, abs(alpha_legacy))``. That scale keeps the comparison proportional to
the magnitude of the legacy value while avoiding a division by a small value.
Circumcenter and weight deviations remain reported diagnostics, and the
candidate must also report convergence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np


METHODS = (
    "fw+bisect",
    "fw+brentq",
    "fw+bisect+newton",
    "fw+brentq+newton",
    "fw+bisect+damped-newton",
    "scipy-slsqp",
    "newton-cold",
)
BASE_SEED = 260_027
INSTANCES_PER_SHAPE = 20
METHOD_TOLERANCES = {
    "fw+bisect": 1e-12,
    "fw+brentq": 1e-12,
    "fw+bisect+newton": 1e-11,
    "fw+brentq+newton": 1e-11,
    "fw+bisect+damped-newton": 1e-11,
    "scipy-slsqp": 1e-6,
    "newton-cold": 1e-12,
}
METHOD_REASONS = {
    "fw+bisect": "52-step bisection line search and relative-gap contract",
    "fw+brentq": "Brent line search and relative-gap contract",
    "fw+bisect+newton": "bisection warm start with Newton polishing",
    "fw+brentq+newton": "Brent warm start with damped Newton polishing",
    "fw+bisect+damped-newton": "bisection warm start with damped Newton polishing",
    "scipy-slsqp": "dual-weight SLSQP with shared gap enforcement and polishing",
    "newton-cold": "uniform-start Newton stability baseline",
}


def _solver(engine: str, source_root: Path | None) -> Callable[..., Any]:
    if engine == "legacy":
        from ellphi_alpha.minimax import solve_minimax
    else:
        if source_root is not None:
            sys.path.insert(0, str(source_root.resolve() / "src"))
        from ellphi._minimax_python import solve_minimax

    return solve_minimax


def _random_instance(k: int, d: int, instance: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(np.random.SeedSequence([BASE_SEED, k, d, instance]))
    matrices = []
    for _ in range(k):
        q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        eigenvalues = np.exp(rng.uniform(-1.0, 1.0, d))
        matrix = (q * eigenvalues) @ q.T
        matrices.append(0.5 * (matrix + matrix.T))
    return np.stack(matrices), rng.standard_normal((k, d))


def make_inputs(output: Path) -> None:
    """Write the exact deterministic inputs shared by both implementations."""
    instances = []
    for k in range(1, 7):
        for d in range(1, 5):
            for instance in range(INSTANCES_PER_SHAPE):
                matrices, centers = _random_instance(k, d, instance)
                instances.append(
                    {
                        "k": k,
                        "d": d,
                        "instance": instance,
                        "matrices": matrices.tolist(),
                        "centers": centers.tolist(),
                    }
                )
    payload = {
        "base_seed": BASE_SEED,
        "instances_per_shape": INSTANCES_PER_SHAPE,
        "instances": instances,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(instances)} inputs to {output}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(
    engine: str,
    revision: str,
    inputs: Path,
    output: Path,
    source_root: Path | None,
) -> None:
    """Generate deterministic solver records for one implementation."""
    solve_minimax = _solver(engine, source_root)
    input_payload = json.loads(inputs.read_text())
    records = []
    for item in input_payload["instances"]:
        matrices = np.asarray(item["matrices"], dtype=float)
        centers = np.asarray(item["centers"], dtype=float)
        for method in METHODS:
            result = solve_minimax(matrices, centers, method=method)
            records.append(
                {
                    "k": item["k"],
                    "d": item["d"],
                    "instance": item["instance"],
                    "method": method,
                    "alpha": result.alpha,
                    "circumcenter": result.circumcenter.tolist(),
                    "weights": result.weights.tolist(),
                    "converged": result.converged,
                }
            )

    payload = {
        "manifest": {
            "engine": engine,
            "revision": revision,
            "source_root": str(source_root.resolve()) if source_root else None,
            "entry_point": "scripts/compare_minimax_engine.py generate",
            "configuration": {
                "methods": METHODS,
                "k": [1, 6],
                "d": [1, 4],
                "instances_per_shape": INSTANCES_PER_SHAPE,
                "comparison_thresholds": METHOD_TOLERANCES,
            },
            "random_seed": BASE_SEED,
            "input_identity": {
                "path": str(inputs),
                "sha256": _sha256(inputs),
                "generator": "SeedSequence([base_seed, k, d, instance])",
            },
            "output": str(output),
            "execution_backend": "local Python/NumPy/SciPy",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "fallback_status": "not-applicable",
            "status": "complete",
        },
        "records": records,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(records)} records to {output}")


def _record_key(record: dict[str, Any]) -> tuple[int, int, int, str]:
    return (
        record["k"],
        record["d"],
        record["instance"],
        record["method"],
    )


def _expected_keys(payload: dict[str, Any]) -> set[tuple[int, int, int, str]]:
    """Return the complete deterministic grid declared by a payload manifest."""
    try:
        configuration = payload["manifest"]["configuration"]
        methods = tuple(configuration["methods"])
        k_min, k_max = configuration["k"]
        d_min, d_max = configuration["d"]
        instances_per_shape = configuration["instances_per_shape"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("payload manifest does not declare a complete grid") from exc
    return {
        (k, d, instance, method)
        for k in range(k_min, k_max + 1)
        for d in range(d_min, d_max + 1)
        for instance in range(instances_per_shape)
        for method in methods
    }


def _records_by_key(
    payload: dict[str, Any], label: str
) -> dict[tuple[int, int, int, str], dict[str, Any]]:
    """Validate uniqueness and completeness before indexing records."""
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"{label} payload has no records list")
    indexed: dict[tuple[int, int, int, str], dict[str, Any]] = {}
    duplicates = []
    for record in records:
        key = _record_key(record)
        if key in indexed:
            duplicates.append(key)
        indexed[key] = record
    if duplicates:
        raise ValueError(f"{label} payload has duplicate record keys: {duplicates[:3]}")
    expected = _expected_keys(payload)
    missing = sorted(expected - indexed.keys())
    extra = sorted(indexed.keys() - expected)
    if missing or extra:
        raise ValueError(
            f"{label} payload has incomplete grid: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    return indexed


def compare(legacy_path: Path, candidate_path: Path) -> int:
    """Compare candidate outputs where the legacy engine converged."""
    legacy_payload = json.loads(legacy_path.read_text())
    candidate_payload = json.loads(candidate_path.read_text())
    legacy = _records_by_key(legacy_payload, "legacy")
    candidate = _records_by_key(candidate_payload, "candidate")
    if legacy.keys() != candidate.keys():
        missing = sorted(legacy.keys() - candidate.keys())
        extra = sorted(candidate.keys() - legacy.keys())
        raise ValueError(
            f"record-key mismatch: missing={missing[:3]}, extra={extra[:3]}"
        )

    converged = 0
    mismatches = []
    strict_tolerance_violations = []
    converged_by_method = {method: 0 for method in METHODS}
    per_method = {
        method: {
            "legacy_converged": 0,
            "mismatches": 0,
            "max_abs_dalpha": 0.0,
            "max_abs_dx": 0.0,
            "max_abs_dmu": 0.0,
            "alpha_le_one": {
                "count": 0,
                "max_abs_deviation": 0.0,
                "max_abs_dalpha": 0.0,
                "max_abs_dx": 0.0,
                "max_abs_dmu": 0.0,
                "tolerance": METHOD_TOLERANCES[method],
            },
            "alpha_gt_one": {
                "count": 0,
                "max_abs_deviation": 0.0,
                "max_relative_deviation": 0.0,
                "max_relative_dalpha": 0.0,
                "max_abs_dalpha": 0.0,
                "max_abs_dx": 0.0,
                "max_abs_dmu": 0.0,
                "tolerance": METHOD_TOLERANCES[method],
            },
            "declared_tolerance": METHOD_TOLERANCES[method],
            **({"reason": METHOD_REASONS[method]} if method in METHOD_REASONS else {}),
        }
        for method in METHODS
    }
    for key, old in legacy.items():
        if not old["converged"]:
            continue
        converged += 1
        method = key[-1]
        converged_by_method[method] += 1
        stats = per_method[method]
        stats["legacy_converged"] += 1
        new = candidate[key]
        dalpha = abs(float(new["alpha"] - old["alpha"]))
        dx = float(
            np.max(
                np.abs(
                    np.asarray(new["circumcenter"]) - np.asarray(old["circumcenter"])
                )
            )
        )
        dmu = float(
            np.max(np.abs(np.asarray(new["weights"]) - np.asarray(old["weights"])))
        )
        stats["max_abs_dalpha"] = max(stats["max_abs_dalpha"], dalpha)
        stats["max_abs_dx"] = max(stats["max_abs_dx"], dx)
        stats["max_abs_dmu"] = max(stats["max_abs_dmu"], dmu)
        deviation = max(dalpha, dx, dmu)
        alpha_scale = max(1.0, abs(float(old["alpha"])))
        if old["alpha"] <= 1:
            partition = stats["alpha_le_one"]
            tolerance = METHOD_TOLERANCES[method]
            if dalpha > tolerance:
                strict_tolerance_violations.append(
                    {
                        "record": key,
                        "alpha_legacy": old["alpha"],
                        "tolerance": tolerance,
                        "dalpha": dalpha,
                        "dx": dx,
                        "dmu": dmu,
                    }
                )
        else:
            partition = stats["alpha_gt_one"]
            tolerance = METHOD_TOLERANCES[method]
            tolerance = max(tolerance, 1e-9 * alpha_scale)
            partition["max_relative_deviation"] = max(
                partition["max_relative_deviation"], deviation / alpha_scale
            )
            partition["max_relative_dalpha"] = max(
                partition["max_relative_dalpha"], dalpha / alpha_scale
            )
        partition["count"] += 1
        partition["max_abs_deviation"] = max(partition["max_abs_deviation"], deviation)
        partition["max_abs_dalpha"] = max(partition["max_abs_dalpha"], dalpha)
        partition["max_abs_dx"] = max(partition["max_abs_dx"], dx)
        partition["max_abs_dmu"] = max(partition["max_abs_dmu"], dmu)
        partition["tolerance"] = tolerance
        same = new["converged"] and dalpha <= tolerance
        if not same:
            stats["mismatches"] += 1
            mismatches.append(key)

    summary = {
        "records_total": len(legacy),
        "records_checked": converged,
        "records_skipped": len(legacy) - converged,
        "records_compared": converged,
        "legacy_converged": converged,
        "legacy_converged_by_method": converged_by_method,
        "mismatches": len(mismatches),
        "first_mismatches": mismatches[:10],
        "strict_tolerance_violations": strict_tolerance_violations[:10],
        "per_method": per_method,
        "status": (
            "complete"
            if converged and not mismatches and not strict_tolerance_violations
            else "failed"
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if not converged or mismatches or strict_tolerance_violations else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    inputs_parser = subparsers.add_parser("inputs")
    inputs_parser.add_argument("--output", type=Path, required=True)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument(
        "--engine", choices=("legacy", "candidate"), required=True
    )
    generate_parser.add_argument("--revision", required=True)
    generate_parser.add_argument("--source-root", type=Path)
    generate_parser.add_argument("--inputs", type=Path, required=True)
    generate_parser.add_argument("--output", type=Path, required=True)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("legacy", type=Path)
    compare_parser.add_argument("candidate", type=Path)

    args = parser.parse_args()
    if args.command == "inputs":
        make_inputs(args.output)
        return 0
    if args.command == "generate":
        generate(
            args.engine,
            args.revision,
            args.inputs,
            args.output,
            args.source_root,
        )
        return 0
    return compare(args.legacy, args.candidate)


if __name__ == "__main__":
    sys.exit(main())
