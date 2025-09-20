"""Poetry build script that ensures the C++ backend is compiled."""

from __future__ import annotations

from build_helpers import compile_cpp_backend


def build() -> None:
    """Compile the optional C++ backend before packaging."""
    compile_cpp_backend(force=True)


if __name__ == "__main__":  # pragma: no cover - manual execution hook
    build()
