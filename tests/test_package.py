"""Smoke tests that verify the package layout is importable end-to-end."""

from __future__ import annotations

import importlib

import pytest


def test_top_level_imports():
    pkg = importlib.import_module("carmen_analysis")
    assert hasattr(pkg, "__version__")
    assert pkg.__version__
    assert hasattr(pkg, "run_pipeline")
    assert hasattr(pkg, "PipelineResult")
    assert hasattr(pkg, "CarmenAnalysisError")


@pytest.mark.parametrize(
    "modpath",
    [
        "carmen_analysis.cli",
        "carmen_analysis.errors",
        "carmen_analysis.pipeline",
        "carmen_analysis.io.matcher",
        "carmen_analysis.io.reader",
        "carmen_analysis.process.binary_results",
        "carmen_analysis.process.median_frame",
        "carmen_analysis.process.norm",
        "carmen_analysis.process.ntc_con_check",
        "carmen_analysis.process.ntcnorm",
        "carmen_analysis.process.summary",
        "carmen_analysis.qc.assay_qc_score",
        "carmen_analysis.qc.flags",
        "carmen_analysis.qc.qual_checks",
        "carmen_analysis.qc.threshold",
        "carmen_analysis.redcap.builder",
        "carmen_analysis.viz.plotting",
        "carmen_analysis.viz.t13_plotting",
    ],
)
def test_submodule_importable(modpath):
    importlib.import_module(modpath)


def test_resource_pdf_present():
    from importlib import resources

    pdf = resources.files("carmen_analysis.resources").joinpath(
        "Assay-Level QC Test Explanation.pdf"
    )
    assert pdf.is_file()
