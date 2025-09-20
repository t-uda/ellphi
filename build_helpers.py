"""Helper utilities for building the packaged C++ backend."""

from __future__ import annotations

import sys
import sysconfig
from pathlib import Path

import os
import shlex
import shutil
import subprocess

PROJECT_ROOT = Path(__file__).parent.resolve()
SRC_ROOT = PROJECT_ROOT / "src"
PACKAGE_ROOT = SRC_ROOT / "ellphi"
SOURCE_FILE = PACKAGE_ROOT / "_tangency_cpp_impl.cpp"


def _compiler_command() -> list[str]:
    env_cxx = os.environ.get("CXX")
    if env_cxx:
        return shlex.split(env_cxx)

    cfg_cxx = sysconfig.get_config_var("CXX")
    if isinstance(cfg_cxx, str) and cfg_cxx:
        return shlex.split(cfg_cxx)

    for candidate in ("c++", "g++", "clang++"):
        if shutil.which(candidate):
            return [candidate]

    if sys.platform.startswith("win"):
        cl = shutil.which("cl")
        if cl:
            return [cl]

    msg = (
        "Unable to locate a C++ compiler. Set the CXX environment variable to override."
    )
    raise RuntimeError(msg)


def shared_library_path() -> Path:
    """Return the expected path to the compiled shared library."""
    suffix = sysconfig.get_config_var("SHLIB_SUFFIX") or ".so"
    return SOURCE_FILE.with_suffix(suffix)


def remove_compiled_backend() -> None:
    """Delete the compiled backend if it exists."""
    library = shared_library_path()
    if library.exists():
        library.unlink()


def compile_cpp_backend(force: bool = False) -> Path:
    """Compile the optional C++ backend and return the shared library path."""
    if not SOURCE_FILE.exists():
        msg = f"Missing C++ source file: {SOURCE_FILE}"
        raise FileNotFoundError(msg)

    library = shared_library_path()
    if (
        library.exists()
        and not force
        and library.stat().st_mtime >= SOURCE_FILE.stat().st_mtime
    ):
        return library

    command = _compiler_command()
    is_msvc = command and Path(command[0]).name.lower() == "cl"

    library.parent.mkdir(parents=True, exist_ok=True)

    if is_msvc:
        output_flag = f"/Fe{library}"
        cmd = [
            *command,
            "/std:c++17",
            "/O2",
            "/LD",
            output_flag,
            str(SOURCE_FILE),
        ]
    else:
        if sys.platform == "darwin":
            cmd = [
                *command,
                "-std=c++17",
                "-O3",
                "-dynamiclib",
                str(SOURCE_FILE),
                "-o",
                str(library),
                "-undefined",
                "dynamic_lookup",
            ]
        else:
            cmd = [
                *command,
                "-std=c++17",
                "-O3",
                "-shared",
                "-fPIC",
                str(SOURCE_FILE),
                "-o",
                str(library),
            ]

    try:
        subprocess.run(cmd, check=True, cwd=str(PACKAGE_ROOT))
    except FileNotFoundError as exc:  # pragma: no cover - build environment issue
        msg = "C++ compiler is required to build the C++ backend"
        raise RuntimeError(msg) from exc
    except subprocess.CalledProcessError as exc:  # pragma: no cover - build failure
        msg = "Failed to compile the C++ tangency backend"
        raise RuntimeError(msg) from exc

    return library


__all__ = [
    "PROJECT_ROOT",
    "SRC_ROOT",
    "PACKAGE_ROOT",
    "SOURCE_FILE",
    "shared_library_path",
    "remove_compiled_backend",
    "compile_cpp_backend",
]
