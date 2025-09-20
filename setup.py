from __future__ import annotations

"""Setuptools configuration with a custom C++ build step."""

import sys
import sysconfig
from pathlib import Path

from setuptools import setup
from setuptools._distutils import ccompiler as distutils_ccompiler
from setuptools._distutils.errors import (
    CompileError,
    DistutilsExecError,
    LinkError,
)
from setuptools._distutils.sysconfig import customize_compiler
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

    compiler = distutils_ccompiler.new_compiler()
    customize_compiler(compiler)

    build_temp = PROJECT_ROOT / "build" / "temp"
    build_temp.mkdir(parents=True, exist_ok=True)

    extra_compile_args: list[str]
    extra_link_args: list[str] = []

    if compiler.compiler_type == "msvc":
        extra_compile_args = ["/std:c++17", "/O2"]
        extra_link_args = ["/LD"]
    else:
        extra_compile_args = ["-std=c++17", "-O3"]
        if sys.platform == "darwin":
            raw_linker = getattr(compiler, "linker_so", [])
            if isinstance(raw_linker, str):
                linker_so = raw_linker.split()
            else:
                linker_so = list(raw_linker)
            linker_so = [arg for arg in linker_so if arg not in {"-bundle", "-shared"}]
            if "-dynamiclib" not in linker_so:
                linker_so.insert(1, "-dynamiclib")
            if "-undefined" not in linker_so:
                linker_so.extend(["-undefined", "dynamic_lookup"])
            compiler.linker_so = linker_so
        else:
            extra_compile_args.append("-fPIC")
            extra_link_args = ["-shared"]

    try:
        objects = compiler.compile(
            [str(SOURCE_FILE)],
            output_dir=str(build_temp),
            extra_postargs=extra_compile_args,
        )
        compiler.link_shared_object(
            objects,
            str(library),
            extra_postargs=extra_link_args,
        )
    except FileNotFoundError as exc:  # pragma: no cover - build environment issue
        msg = "C++ compiler is required to build the C++ backend"
        raise RuntimeError(msg) from exc
    except (
        CompileError,
        LinkError,
        DistutilsExecError,
    ) as exc:  # pragma: no cover - build failure
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
