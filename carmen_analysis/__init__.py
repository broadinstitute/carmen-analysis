"""CARMEN analysis pipeline — Cas13-based multiplexed viral diagnostic data processing."""

__version__ = "5.5.0"

from carmen_analysis.errors import CarmenAnalysisError
from carmen_analysis.pipeline import PipelineResult, run_pipeline

__all__ = ["__version__", "CarmenAnalysisError", "PipelineResult", "run_pipeline"]
