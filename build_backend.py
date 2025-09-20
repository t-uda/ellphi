"""Custom PEP 517 backend that ensures the C++ library is built."""

from __future__ import annotations

from typing import Any

from poetry.core.masonry.api import (
    build_sdist as _poetry_build_sdist,
    build_wheel as _poetry_build_wheel,
    prepare_metadata_for_build_wheel as _poetry_prepare_metadata,
)

from build_helpers import compile_cpp_backend, remove_compiled_backend


def build_wheel(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    compile_cpp_backend()
    return _poetry_build_wheel(wheel_directory, config_settings, metadata_directory)


def prepare_metadata_for_build_wheel(
    metadata_directory: str,
    config_settings: dict[str, Any] | None = None,
) -> str:
    compile_cpp_backend()
    return _poetry_prepare_metadata(metadata_directory, config_settings)


def build_sdist(directory: str, config_settings: dict[str, Any] | None = None) -> str:
    remove_compiled_backend()
    return _poetry_build_sdist(directory, config_settings)
