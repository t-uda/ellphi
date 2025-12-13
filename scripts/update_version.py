#!/usr/bin/env python3
"""Synchronize project version in packaging metadata and the runtime module.

Usage:
    poetry run python scripts/update_version.py 1.2.3

The script validates the provided version against PEP 440, normalizes it to the
canonical form, and writes the value to both `pyproject.toml` and
`src/ellphi/_version.py`. Use it during releases to keep the distributed
metadata and the runtime-facing `__version__` constant aligned.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from packaging.version import InvalidVersion, Version

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
VERSION_MODULE = ROOT / "src" / "ellphi" / "_version.py"


def write_version_module(version: str) -> None:
    VERSION_MODULE.write_text(
        "\n".join(
            [
                '"""Canonical package version."""',
                "",
                f'__version__ = "{version}"',
                "",
            ]
        ),
        encoding="utf-8",
    )


def update_pyproject_version(version: str) -> None:
    content = PYPROJECT.read_text(encoding="utf-8")
    new_content, count = re.subn(
        r'(?m)^version\s*=\s*"[^"]+"', f'version = "{version}"', content, count=1
    )
    if count == 0:
        raise ValueError("Unable to locate [project] version field in pyproject.toml")
    PYPROJECT.write_text(new_content, encoding="utf-8")


def validate_version(raw_version: str) -> str:
    try:
        parsed = Version(raw_version)
    except InvalidVersion as exc:  # pragma: no cover - argparse handles display
        raise argparse.ArgumentTypeError(
            f"{raw_version!r} is not a valid PEP 440 version"
        ) from exc
    return parsed.public


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "version",
        type=validate_version,
        help="PEP 440 version to record; normalized before writing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    update_pyproject_version(args.version)
    write_version_module(args.version)


if __name__ == "__main__":
    main()
