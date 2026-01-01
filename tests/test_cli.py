"""Tests for the ellphi CLI."""

import unittest.mock
from contextlib import redirect_stdout
import io
import pytest

from ellphi import _main, version_info


def test_cli_version():
    """Test the --version argument."""
    with unittest.mock.patch("sys.argv", ["ellphi", "--version"]):
        with io.StringIO() as buf, redirect_stdout(buf):
            _main()
            output = buf.getvalue().strip()
    assert output == version_info()


def test_cli_build_info():
    """Test the --build-info argument."""
    with unittest.mock.patch("sys.argv", ["ellphi", "--build-info"]):
        with io.StringIO() as buf, redirect_stdout(buf):
            _main()
            output = buf.getvalue().strip().splitlines()
    assert len(output) == 1
    assert output[0].startswith("BuildInfo(")


def test_cli_version_and_build_info():
    """Test --version and --build-info together."""
    with unittest.mock.patch("sys.argv", ["ellphi", "--version", "--build-info"]):
        with io.StringIO() as buf, redirect_stdout(buf):
            _main()
            output = buf.getvalue().strip().splitlines()
    assert output[0] == version_info()
    assert output[1].startswith("BuildInfo(")


def test_cli_help():
    """Test the --help argument."""
    with unittest.mock.patch("sys.argv", ["ellphi", "--help"]):
        with io.StringIO() as buf, redirect_stdout(buf):
            with pytest.raises(SystemExit):
                _main()
            output = buf.getvalue()
    assert "usage: ellphi" in output


def test_cli_no_args():
    """Test running with no arguments."""
    with unittest.mock.patch("sys.argv", ["ellphi"]):
        with io.StringIO() as buf, redirect_stdout(buf):
            _main()
            output = buf.getvalue()
    assert "usage: ellphi" in output
