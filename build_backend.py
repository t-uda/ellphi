"""Custom PEP 517 backend that builds the ellphi package without third-party tooling."""

from __future__ import annotations

import base64
import csv
import hashlib
import re
import shutil
import sysconfig
import tarfile
import tempfile
from pathlib import Path
from typing import Iterable
from zipfile import ZIP_DEFLATED, ZipFile

import tomllib

from build_helpers import (
    PACKAGE_ROOT,
    PROJECT_ROOT,
    compile_cpp_backend,
    remove_compiled_backend,
)


def _read_pyproject() -> dict:
    return tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _project_metadata() -> dict:
    data = _read_pyproject()["project"]
    dependencies = list(data.get("dependencies", []))
    return {
        "name": data["name"],
        "version": data["version"],
        "summary": data.get("description", ""),
        "requires_python": data.get("requires-python"),
        "dependencies": dependencies,
    }


def _normalise_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "_", name).lower()


def _write_text(path: Path, lines: Iterable[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _wheel_tag() -> str:
    platform = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    return f"py3-none-{platform}"


def _copy_package(destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for path in PACKAGE_ROOT.rglob("*"):
        if path.name == "__pycache__" or path.suffix == ".pyc":
            continue
        rel = path.relative_to(PACKAGE_ROOT)
        target = destination / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def _record_rows(root: Path) -> list[tuple[str, str, str]]:
    records: list[tuple[str, str, str]] = []
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(root).as_posix()
        if rel.endswith("RECORD"):
            continue
        data = path.read_bytes()
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode("ascii")
        digest = digest.rstrip("=")
        records.append((rel, f"sha256={digest}", str(len(data))))
    return records


def build_wheel(
    wheel_directory: str,
    config_settings: dict | None = None,
    metadata_directory: str | None = None,
) -> str:
    del config_settings, metadata_directory

    meta = _project_metadata()
    compile_cpp_backend(force=True)

    distribution = _normalise_distribution(meta["name"])
    tag = _wheel_tag()
    wheel_name = f"{distribution}-{meta['version']}-{tag}.whl"

    wheel_dir = Path(wheel_directory)
    wheel_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        build_root = Path(tmp)
        package_dest = build_root / "ellphi"
        _copy_package(package_dest)

        dist_info = build_root / f"{distribution}-{meta['version']}.dist-info"
        dist_info.mkdir(parents=True, exist_ok=True)

        metadata_lines = [
            "Metadata-Version: 2.1",
            f"Name: {meta['name']}",
            f"Version: {meta['version']}",
        ]
        if meta["summary"]:
            metadata_lines.append(f"Summary: {meta['summary']}")
        if meta["requires_python"]:
            metadata_lines.append(f"Requires-Python: {meta['requires_python']}")
        for dep in meta["dependencies"]:
            metadata_lines.append(f"Requires-Dist: {dep}")
        _write_text(dist_info / "METADATA", metadata_lines)

        wheel_lines = [
            "Wheel-Version: 1.0",
            "Generator: ellphi-build-backend",
            "Root-Is-Purelib: false",
            f"Tag: {tag}",
        ]
        _write_text(dist_info / "WHEEL", wheel_lines)

        (dist_info / "top_level.txt").write_text("ellphi\n", encoding="utf-8")

        record_path = dist_info / "RECORD"
        records = _record_rows(build_root)
        record_rel = record_path.relative_to(build_root).as_posix()
        with record_path.open("w", newline="") as record_file:
            writer = csv.writer(record_file)
            writer.writerows(records)
            writer.writerow([record_rel, "", ""])

        wheel_path = wheel_dir / wheel_name
        with ZipFile(wheel_path, "w", compression=ZIP_DEFLATED) as archive:
            for path in sorted(build_root.rglob("*")):
                if path.is_dir():
                    continue
                rel = path.relative_to(build_root).as_posix()
                archive.write(path, rel)

    return wheel_name


def build_sdist(
    sdist_directory: str,
    config_settings: dict | None = None,
) -> str:
    del config_settings

    meta = _project_metadata()
    remove_compiled_backend()

    distribution = meta["name"]
    archive_name = f"{distribution}-{meta['version']}"
    sdist_dir = Path(sdist_directory)
    sdist_dir.mkdir(parents=True, exist_ok=True)
    target = sdist_dir / f"{archive_name}.tar.gz"

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / archive_name
        staging.mkdir(parents=True, exist_ok=True)

        shutil.copy2(PROJECT_ROOT / "pyproject.toml", staging / "pyproject.toml")
        for filename in [
            "README.md",
            "LICENSE",
            "setup.py",
            "build_backend.py",
            "build_helpers.py",
            "build.py",
        ]:
            source = PROJECT_ROOT / filename
            if source.exists():
                shutil.copy2(source, staging / filename)

        package_dest = staging / "src" / "ellphi"
        _copy_package(package_dest)

        with tarfile.open(target, "w:gz") as tar:
            tar.add(staging, arcname=archive_name)

    return target.name


def prepare_metadata_for_build_wheel(
    metadata_directory: str,
    config_settings: dict | None = None,
) -> str:
    del config_settings

    meta = _project_metadata()
    distribution = _normalise_distribution(meta["name"])
    metadata_root = Path(metadata_directory)
    dist_info = metadata_root / f"{distribution}-{meta['version']}.dist-info"
    dist_info.mkdir(parents=True, exist_ok=True)

    metadata_lines = [
        "Metadata-Version: 2.1",
        f"Name: {meta['name']}",
        f"Version: {meta['version']}",
    ]
    if meta["summary"]:
        metadata_lines.append(f"Summary: {meta['summary']}")
    if meta["requires_python"]:
        metadata_lines.append(f"Requires-Python: {meta['requires_python']}")
    for dep in meta["dependencies"]:
        metadata_lines.append(f"Requires-Dist: {dep}")
    _write_text(dist_info / "METADATA", metadata_lines)

    wheel_lines = [
        "Wheel-Version: 1.0",
        "Generator: ellphi-build-backend",
        "Root-Is-Purelib: false",
        f"Tag: {_wheel_tag()}",
    ]
    _write_text(dist_info / "WHEEL", wheel_lines)

    top_level = dist_info / "top_level.txt"
    top_level.write_text("ellphi\n", encoding="utf-8")

    return dist_info.name


__all__ = [
    "build_wheel",
    "build_sdist",
    "prepare_metadata_for_build_wheel",
]
