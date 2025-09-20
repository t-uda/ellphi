"""Poetry build script that compiles the optional C++ backend."""

from __future__ import annotations

from build_helpers import compile_cpp_backend


def build() -> None:
    """Compile the C++ tangency backend before packaging."""
    compile_cpp_backend()
