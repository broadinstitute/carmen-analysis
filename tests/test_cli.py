"""CLI argument-parsing tests (no pipeline execution)."""

from __future__ import annotations

import pytest

from carmen_analysis import cli


def test_help_runs_without_error(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "carmen-analyze" in out


def test_unknown_panel_rejected(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main([
            "--assignment", "missing.xlsx",
            "--data", "missing.csv",
            "--panel", "not_a_real_panel",
        ])
    # argparse exits with 2 for unknown choices
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "panel" in err.lower()


def test_required_args_enforced(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code == 2
