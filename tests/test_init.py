"""Tests for the :mod:`ellphi` package initialisation."""

import importlib
import sys
from importlib.metadata import PackageNotFoundError

from ellphi._version import __version__ as _CANONICAL_VERSION


def test_import_without_package_metadata(monkeypatch):
    """The package should fall back to a sensible version when metadata is missing."""

    def _raise_version(_name):
        raise PackageNotFoundError

    monkeypatch.setattr("importlib.metadata.version", _raise_version)

    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "ellphi" or name.startswith("ellphi.")
    }

    for name in list(saved_modules):
        sys.modules.pop(name, None)

    try:
        module = importlib.import_module("ellphi")
        assert module.__version__ == _CANONICAL_VERSION
    finally:
        for name in [
            key for key in sys.modules if key == "ellphi" or key.startswith("ellphi.")
        ]:
            sys.modules.pop(name, None)

        sys.modules.update(saved_modules)
