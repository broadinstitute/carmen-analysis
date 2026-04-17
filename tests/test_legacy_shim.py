"""Tests for the backwards-compatible analyze_run.py shim.

Verifies argument parsing and error paths without actually running the
pipeline (that would need real fixture data — covered by a separate e2e test).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_shim():
    """Import the root analyze_run.py as a module without polluting sys.modules.

    The shim isn't part of the package, so we have to load it from the repo
    root by file path.
    """
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("_analyze_run_shim", root / "analyze_run.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_help_returns_zero(capsys):
    shim = _load_shim()
    rc = shim.main(["analyze_run.py", "--help"])
    assert rc == 0
    captured = capsys.readouterr()
    # The shim prints its own __doc__ which mentions the legacy invocation
    assert "1.8_Mean" in captured.out or "carmen-analyze" in captured.out


def test_no_args_prints_help(capsys):
    shim = _load_shim()
    rc = shim.main(["analyze_run.py"])
    assert rc == 0


def test_unknown_panel_returns_2(capsys):
    shim = _load_shim()
    rc = shim.main(["analyze_run.py", "bogus_panel"])
    assert rc == 2
    captured = capsys.readouterr()
    assert "bogus_panel" in captured.err


def test_missing_input_files_returns_2(tmp_path, monkeypatch, capsys):
    """A valid panel + an empty cwd should fail cleanly with exit code 2."""
    monkeypatch.chdir(tmp_path)
    shim = _load_shim()
    rc = shim.main(["analyze_run.py", "1.8_Mean"])
    assert rc == 2
    captured = capsys.readouterr()
    assert "No assignment XLSX" in captured.err


def test_redcap_flag_recognized(monkeypatch, tmp_path, capsys):
    """The third arg 'REDCAP' should be parsed as the redcap flag.

    The shim still fails because there are no input files in tmp_path, but
    that's the *next* error — argument parsing should pass.
    """
    monkeypatch.chdir(tmp_path)
    shim = _load_shim()
    rc = shim.main(["analyze_run.py", "1.8_Mean", "REDCAP"])
    # Same exit-code (2) as missing-files case; what we're checking here is
    # that the third arg didn't trip the panel validator.
    assert rc == 2
    err = capsys.readouterr().err
    assert "bogus" not in err.lower() and "unknown panel" not in err.lower()


@pytest.fixture(autouse=True)
def _cleanup_shim_module():
    yield
    sys.modules.pop("_analyze_run_shim", None)
