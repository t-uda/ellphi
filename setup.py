from __future__ import annotations

"""Setuptools configuration with a custom C++ build step."""

import os
import subprocess
import sysconfig
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as build_py_orig
from setuptools.command.develop import develop as develop_orig
from setuptools.command.sdist import sdist as sdist_orig

PROJECT_ROOT = Path(__file__).parent
SOURCE_FILE = PROJECT_ROOT / "src" / "ellphi" / "_tangency_cpp_impl.cpp"


def _shared_library_path() -> Path:
    suffix = sysconfig.get_config_var("SHLIB_SUFFIX") or ".so"
    return SOURCE_FILE.with_suffix(suffix)


def _compile_cpp_backend(force: bool = False) -> Path:
    if not SOURCE_FILE.exists():
        msg = f"Missing C++ source file: {SOURCE_FILE}"
        raise FileNotFoundError(msg)

    library = _shared_library_path()

    if (
        not force
        and library.exists()
        and library.stat().st_mtime >= SOURCE_FILE.stat().st_mtime
    ):
        return library

    library.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "g++",
        "-std=c++17",
        "-O3",
        "-shared",
        str(SOURCE_FILE),
        "-o",
        str(library),
    ]
    if os.name != "nt":
        cmd.insert(4, "-fPIC")

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:  # pragma: no cover - build environment issue
        msg = "g++ compiler is required to build the C++ backend"
        raise RuntimeError(msg) from exc
    except subprocess.CalledProcessError as exc:  # pragma: no cover - build failure
        msg = "Failed to compile the C++ tangency backend"
        raise RuntimeError(msg) from exc

    return library


def _remove_compiled_backend() -> None:
    library = _shared_library_path()
    if library.exists():
        library.unlink()


class build_py(build_py_orig):
    def run(self) -> None:  # pragma: no cover - executed during packaging
        _compile_cpp_backend(force=self.force)
        super().run()


class develop(develop_orig):
    def run(self) -> None:  # pragma: no cover - executed during editable installs
        _compile_cpp_backend(force=True)
        super().run()


class sdist(sdist_orig):
    def run(self) -> None:  # pragma: no cover - executed during packaging
        try:
            _remove_compiled_backend()
        finally:
            super().run()


setup(
    cmdclass={
        "build_py": build_py,
        "develop": develop,
        "sdist": sdist,
    },
)
