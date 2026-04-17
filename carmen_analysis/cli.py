"""Command-line entry point for the CARMEN analysis pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from carmen_analysis import __version__
from carmen_analysis.errors import CarmenAnalysisError
from carmen_analysis.pipeline import SUPPORTED_PANELS, run_pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="carmen-analyze",
        description=(
            "Run the CARMEN analysis pipeline on a Biomark IFC export. "
            "Produces hit calls, NTC-normalized quantitative results, QC reports, "
            "and per-timepoint heatmaps."
        ),
    )
    parser.add_argument(
        "--assignment", "-a", required=True, type=Path,
        help="Path to the assignment-sheet XLSX.",
    )
    parser.add_argument(
        "--data", "-d", required=True, type=Path,
        help="Path to the Biomark instrument CSV.",
    )
    parser.add_argument(
        "--panel", "-p", required=True, choices=SUPPORTED_PANELS,
        help="Thresholding mode: '1.8_Mean' (RVP) or '3_SD' (BBP).",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=Path.cwd(),
        help="Directory under which the per-run output folder is created (default: cwd).",
    )
    parser.add_argument(
        "--barcode", default=None,
        help="Override IFC barcode label (default: derived from assignment filename).",
    )
    parser.add_argument(
        "--redcap", action="store_true",
        help="Also produce the REDCAP-formatted export CSV.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_parser().parse_args(argv)
    try:
        result = run_pipeline(
            assignment_xlsx=args.assignment,
            data_csv=args.data,
            panel=args.panel,
            output_dir=args.output,
            redcap=args.redcap,
            barcode=args.barcode,
            progress=print,
        )
    except CarmenAnalysisError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Done. Outputs in: {result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
