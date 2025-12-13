"""Command-line entry point for :mod:`ellphi`.

Allows executing ``python -m ellphi`` to access the minimal CLI that exposes
version information.
"""

from . import _main


if __name__ == "__main__":  # pragma: no cover - thin wrapper
    _main()
