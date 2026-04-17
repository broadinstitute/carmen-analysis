# Agent guide for carmen-analysis

This file briefs AI coding agents (Claude Code, Cursor, OpenAI agents, etc.)
on the structure of this repo and the conventions to follow when editing it.
It is the source of truth; `CLAUDE.md` re-exports it.

## What this project is

A Cas13-based viral diagnostic pipeline that processes Standard BioTools
Biomark fluorescence exports into hit calls, QC reports, and heatmaps. The
audience is wet-lab clinicians, bioinformaticians, and a long-tail of REDCap
consumers downstream.

## Architecture in one screen

```
analyze_run.py                 # legacy shim; cwd-globs and delegates to run_pipeline
carmen_analysis/
├── __init__.py                # re-exports run_pipeline, PipelineResult, CarmenAnalysisError
├── cli.py                     # argparse `carmen-analyze` entry point
├── errors.py                  # CarmenAnalysisError (the only library-level exception)
├── pipeline.py                # `run_pipeline(...)` — the orchestration, do not duplicate
├── io/
│   ├── reader.py              # parses the Biomark CSV into per-phrase DataFrames
│   └── matcher.py             # joins assignment XLSX → assays/samples
├── process/                   # numerical pipeline stages (norm, ntcnorm, median_frame, ...)
├── qc/                        # threshold, qual_checks, flags, assay_qc_score
├── viz/                       # plotting + t13_plotting (matplotlib/seaborn heatmaps)
├── redcap/builder.py          # REDCap CSV emission
├── resources/                 # bundled non-code data (e.g. QC explanation PDF)
└── web/app.py                 # Streamlit UI; stateless, calls run_pipeline
                                #   exposes `run()` for the carmen-web entry point
```

The orchestration is **only** in `pipeline.py`. Sub-modules do not import each
other; they're independent stages that `pipeline.py` wires together.

## Console scripts

- `carmen-analyze` → `carmen_analysis.cli:main` — argparse CLI.
- `carmen-web`     → `carmen_analysis.web.app:run` — launches the Streamlit UI.

## Conventions

- **Python**: 3.12 only. `from __future__ import annotations` everywhere.
- **Errors at boundaries**: raise `CarmenAnalysisError` for any condition the
  caller could plausibly recover from (bad input, missing file, unknown
  panel). Do not call `sys.exit()` from library code.
- **No globals**: take inputs as parameters; never read `sys.argv` or `cwd`
  inside the package. The legacy `analyze_run.py` shim is the only place
  those legacy behaviors live.
- **I/O flexibility**: `run_pipeline` and the readers accept both paths and
  binary file-like objects (the web UI streams uploads).
- **Output layout**: preserved from the legacy script —
  `output_{barcode}_{panel}_v{version}/{RESULTS,QUALITY_CONTROL,R&D}_{barcode}/`.
  Don't reshape it; downstream automation depends on these names.
- **Versioning**: bumped in three places that must stay in sync —
  `carmen_analysis/__init__.py`, `pyproject.toml`, and any `v*` git tag.
- **No PHI in the repo**: `*.csv` and `*.xlsx` are gitignored; only fixtures
  under `tests/fixtures/` may be checked in (and only with synthetic data).

## Adding a feature

1. If it's a new pipeline stage, add a module under `process/` or `qc/`.
2. Wire it into `carmen_analysis/pipeline.py` only — the order matters.
3. If it's user-facing, expose an argument on `cli.py` and a control on
   `web/app.py`.
4. Add a smoke test in `tests/`.

## Running tests / lint

```bash
pip install -e ".[web,dev]"
pytest -q
ruff check carmen_analysis tests
carmen-analyze --help     # CLI smoke test
```

CI (`.github/workflows/ci.yml`) enforces all four on every PR.

Legacy modules (everything moved by `git mv` from the repo root into
`io/`, `process/`, `qc/`, `viz/`, `redcap/`) carry per-file ruff ignores in
`pyproject.toml` for cosmetic findings (whitespace, loop variables, zip
strict). Don't churn those modules to clear the ignores in unrelated PRs.
The new code (`pipeline.py`, `cli.py`, `errors.py`, `web/app.py`,
`__init__.py`) has no per-file ignores and must stay clean.

## Build / deploy

- `.github/workflows/docker.yml` builds the image, runs Trivy
  (HIGH/CRITICAL, fail on findings), pushes to
  `us-central1-docker.pkg.dev/sabeti-adapt/carmen-analysis/carmen-analysis`
  via Workload Identity Federation. No long-lived service-account keys.
- `.github/workflows/release.yml` builds sdist+wheel on `v*` tags and creates
  the GitHub Release.
- `terraform/` provisions the Cloud Run service, HTTPS LB, IAP, and the
  GCP-managed SSL certificate. The static IP `carmen-analysis-ip` already
  exists in the `sabeti-adapt` project; the BITS DNS ticket points
  `carmen-analysis.broadinstitute.org` at it.

## Things to avoid

- Don't duplicate orchestration logic across `cli.py`, `web/app.py`, and the
  legacy shim. They all funnel through `run_pipeline`.
- Don't add print statements in library code; use the `progress` callback or
  a module-level `logging.getLogger(__name__)`.
- Don't bake user data into the Docker image. `.dockerignore` excludes
  `*.csv` / `*.xlsx`; keep it that way.
- Don't introduce backwards-incompatible CLI changes without updating the
  legacy shim and noting it in the release notes.

## Where to look first

- Adding a CLI flag: `carmen_analysis/cli.py` and `carmen_analysis/web/app.py`.
- Changing an output filename or sheet name: `carmen_analysis/pipeline.py`.
- Tweaking thresholding math: `carmen_analysis/qc/threshold.py`.
- Adjusting plot styling: `carmen_analysis/viz/`.
- Touching infra/deploy: `terraform/` and `.github/workflows/docker.yml`.
