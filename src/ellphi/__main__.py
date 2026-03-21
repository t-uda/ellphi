"""Command-line entry point for `ellphi`.

Allows executing ``python -m ellphi`` to access the minimal CLI that exposes
version/build information.
"""

from . import _main


if __name__ == "__main__":  # pragma: no cover - thin wrapper
    _main()
