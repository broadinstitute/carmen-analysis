"""Legacy shim for the historical CARMEN entry point.

Preserves the old invocation:

    python3 analyze_run.py 1.8_Mean              # RVP-style thresholding
    python3 analyze_run.py 3_SD                  # BBP-style thresholding
    python3 analyze_run.py 1.8_Mean REDCAP       # also emit REDCAP CSV

It globs the current working directory for the assignment XLSX and the
instrument CSV, then delegates to :func:`carmen_analysis.pipeline.run_pipeline`.

For new usage prefer the proper CLI: ``carmen-analyze --help``.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

from carmen_analysis.errors import CarmenAnalysisError
from carmen_analysis.pipeline import SUPPORTED_PANELS, run_pipeline


def _pick_one(pattern: str, kind: str) -> Path:
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise CarmenAnalysisError(f"No {kind} file matching {pattern!r} in current directory.")
    if len(matches) > 1:
        raise CarmenAnalysisError(
            f"Multiple {kind} files matching {pattern!r} in current directory: {matches}. "
            "Move the unwanted ones out, or use the new CLI: carmen-analyze --help."
        )
    return Path(matches[0])


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] in {"-h", "--help"}:
        print(__doc__)
        return 0

    panel = argv[1]
    if panel not in SUPPORTED_PANELS:
        print(
            f"error: unknown panel {panel!r}. Supported: {SUPPORTED_PANELS}",
            file=sys.stderr,
        )
        return 2
    redcap = len(argv) >= 3 and argv[2].upper() == "REDCAP"

    try:
        assignment = _pick_one("*.xlsx", "assignment XLSX")
        data = _pick_one("*.csv", "instrument CSV")
        run_pipeline(
            assignment_xlsx=assignment,
            data_csv=data,
            panel=panel,
            output_dir=Path.cwd(),
            redcap=redcap,
            progress=print,
        )
    except CarmenAnalysisError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
