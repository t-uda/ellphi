"""Backend selection helpers for tangency computations."""

from __future__ import annotations

import os
from types import ModuleType
from typing import Literal

from . import _tangency_cpp

BackendName = Literal["python", "cpp"]

__all__ = [
    "BackendName",
    "available_backends",
    "get_backend",
    "set_backend",
    "resolve_backend",
    "has_cpp_backend",
    "require_cpp_backend",
]

_PREFERENCE = os.environ.get("ELLPHI_BACKEND", "auto").strip().lower()
if _PREFERENCE not in {"auto", "python", "cpp"}:
    _PREFERENCE = "auto"

_ACTIVE_BACKEND: BackendName | None = None


def available_backends() -> tuple[BackendName, ...]:
    """Return the tuple of recognised backend names."""
    return ("python", "cpp")


def _ensure_cpp() -> ModuleType | None:
    """Attempt to load the C++ backend and return the module on success."""
    return _tangency_cpp.get_module()


def has_cpp_backend() -> bool:
    """Return ``True`` when the compiled backend can be used."""
    return _ensure_cpp() is not None


def require_cpp_backend() -> ModuleType:
    """Return the compiled backend module or raise if unavailable."""
    module = _ensure_cpp()
    if module is None:
        error = _tangency_cpp.get_error()
        if error is None:
            raise RuntimeError("C++ backend is not available")
        raise RuntimeError("Failed to load C++ backend") from error
    return module


def resolve_backend(preference: str | None) -> BackendName:
    """Resolve a backend name from user preference."""
    pref = (preference or _PREFERENCE).strip().lower()
    if pref not in {"auto", "python", "cpp"}:
        raise ValueError(f"Unknown backend '{preference}'")
    if pref == "python":
        return "python"
    if pref == "cpp":
        # Explicit request: raise on failure to make debugging clear.
        require_cpp_backend()
        return "cpp"
    # Automatic selection prefers the compiled backend but falls back gracefully.
    return "cpp" if has_cpp_backend() else "python"


def get_backend() -> BackendName:
    """Return the currently active backend, initialising on first use."""
    global _ACTIVE_BACKEND
    if _ACTIVE_BACKEND is None:
        _ACTIVE_BACKEND = resolve_backend(None)
    return _ACTIVE_BACKEND


def set_backend(name: BackendName) -> None:
    """Update the globally active backend."""
    global _ACTIVE_BACKEND
    backend = resolve_backend(name)
    _ACTIVE_BACKEND = backend
