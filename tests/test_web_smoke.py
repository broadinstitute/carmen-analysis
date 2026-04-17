"""Import-time smoke test for the Streamlit web app module.

This catches import-time errors (typos in module-level code, missing
dependencies, broken cross-package imports) without needing a real Streamlit
runtime. Streamlit itself must be installed (`pip install -e .[web]`).
"""

from __future__ import annotations

import importlib

import pytest


def test_web_app_module_imports():
    pytest.importorskip("streamlit")
    mod = importlib.import_module("carmen_analysis.web.app")
    # The two callables wired up to entry points + invoked by Streamlit.
    assert callable(mod.render)
    assert callable(mod.run)
    # Internal helpers should also be present.
    assert callable(mod._zip_directory)
    assert callable(mod._df_to_tsv_bytes)


def test_panel_choices_cover_all_supported_modes():
    pytest.importorskip("streamlit")
    mod = importlib.import_module("carmen_analysis.web.app")
    from carmen_analysis.pipeline import SUPPORTED_PANELS

    flat: set[str] = set()
    for choice in mod._PANEL_CHOICES:
        flat.update(choice.panels)
    assert flat == set(SUPPORTED_PANELS)


def test_panel_choices_include_run_both():
    pytest.importorskip("streamlit")
    mod = importlib.import_module("carmen_analysis.web.app")
    has_multi = any(len(c.panels) > 1 for c in mod._PANEL_CHOICES)
    assert has_multi, "expected a 'run both and compare' option"
