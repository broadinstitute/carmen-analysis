"""Pipeline-level exceptions surfaced to CLI/web callers."""


class CarmenAnalysisError(Exception):
    """Raised when the analysis pipeline cannot proceed (bad input, missing files, etc)."""
