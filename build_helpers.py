from __future__ import annotations

"""Shared helpers for building the compiled C++ backend."""

import os
import subprocess
import sysconfig
from pathlib import Path

__all__ = [
    "project_root",
    "shared_library_path",
    "compile_cpp_backend",
    "remove_compiled_backend",
]


def project_root() -> Path:
    return Path(__file__).parent


def shared_library_path() -> Path:
    suffix = sysconfig.get_config_var("SHLIB_SUFFIX") or ".so"
    source = project_root() / "src" / "ellphi" / "_tangency_cpp_impl.cpp"
    return source.with_suffix(suffix)


def compile_cpp_backend(force: bool = False) -> Path:
    source = project_root() / "src" / "ellphi" / "_tangency_cpp_impl.cpp"
    library = shared_library_path()

    if not source.exists():
        raise FileNotFoundError(f"Missing C++ source file: {source}")

    if (
        not force
        and library.exists()
        and library.stat().st_mtime >= source.stat().st_mtime
    ):
        return library

    library.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "g++",
        "-std=c++17",
        "-O3",
        "-shared",
        str(source),
        "-o",
        str(library),
    ]
    if os.name != "nt":
        cmd.insert(4, "-fPIC")

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("g++ compiler is required to build the C++ backend") from exc
    except (
        subprocess.CalledProcessError
    ) as exc:  # pragma: no cover - build time failure
        raise RuntimeError("Failed to compile the C++ tangency backend") from exc

    return library


def remove_compiled_backend() -> None:
    library = shared_library_path()
    if library.exists():
        library.unlink()
