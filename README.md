# carmen-analysis

CARMEN is a Cas13-based, multiplexed viral diagnostic developed in the Sabeti
Lab. After running a Standard BioTools Dynamic Array™ IFC on the Standard
BioTools Biomark™ instrument, this package converts the raw fluorescence
export into hit calls, NTC-normalized quantitative signals, QC reports, and
heatmaps.

The pipeline is pathogen-agnostic and works for any combination of viral
assays validated for CARMEN.

**Software version:** 6.0.0 (also exposed as `carmen_analysis.__version__`).

---

## Three ways to use it

| Audience | Use this |
|---|---|
| Wet-lab tech with two files in hand | The web app (`carmen-analysis.broadinstitute.org`, once the DNS record is live) |
| Bioinformatician analyzing many runs | The `carmen-analyze` CLI |
| Pipeline integrator | `from carmen_analysis import run_pipeline` |

All three call the same `carmen_analysis.pipeline.run_pipeline` function.

---

## Required inputs

You always need exactly two files:

1. **An assignment sheet** — `.xlsx` named `{IFC_Barcode}_{Chip_Dimension}_assignment.xlsx`.
   Templates live in [this Google Drive folder](https://drive.google.com/drive/folders/1iQsmyuwRtDyMgtT2YvgJ4yv_sGcJMhgv?usp=drive_link).
   Enter samples in Sheet 1 and assays in Sheet 2; do **NOT** edit
   the `layout_assays` / `layout_samples` tabs.

2. **A Biomark instrument export** — `.csv` named `{IFC_Barcode}.csv`
   (Standard BioTools' `Results_all.csv`, renamed).

### Required controls in the assignment sheet

Each plate must include:

- **NTC** ("No Target Control") — must contain `NTC` in the name; should be negative.
- **NDC** ("No Detection Control") — must contain `NDC`; should be negative.
- **CPC** ("Combined Positive Control") — must contain `CPC`; should be positive for all assays.
- **no_crRNA** assay — must contain `no_crRNA`; every sample should be negative against it.

Anything not matching `NTC` / `NDC` / `CPC` is treated as a clinical sample.

### Panel-specific assay/sample suffixing

Append `_RVP`, `_P1`, or `_P2` to assay/sample names corresponding to the RVP
panel, BBP Panel #1, or BBP Panel #2. Assays shared across panels (e.g. RNaseP)
may carry multiple suffixes (e.g. `RNaseP_P1_P2`).

### Thresholding mode

Pick the threshold appropriate for your panel:

- `1.8_Mean` — RVP default. Positive if signal ≥ 1.8 × mean(NTC).
- `3_SD` — BBP default. Positive if signal ≥ mean(NTC) + 3 × SD(NTC) per assay.

---

## Web app

Open the deployed web app, drop the two files, pick the panel, click
**Analyze**, download the ZIP. The UI offers:

- A panel selector with friendly labels for RVP and BBP, plus a
  **"Run both and compare"** option that runs both thresholding modes on the
  same dataset and shows the results side-by-side (this is a common workflow
  for the wet-lab team).
- An optional REDCap export.
- A prominent **binary results** table + one-click TSV download, formatted
  for paste-into-GraphPad-Prism (so you don't have to manually replace
  POSITIVE → 1 / NEGATIVE → 0).
- Inline preview of the final NTC-normalized heatmap, with per-timepoint
  heatmaps in a collapsed expander.

No credentials are stored. No data is retained between sessions.

To run the app locally (after `pip install -e ".[web]"` from a clone):

```bash
carmen-web                                       # console script (preferred)

# Or, equivalently:
streamlit run "$(python -c 'import carmen_analysis.web.app as a; print(a.__file__)')"
```

---

## CLI

```bash
# From a clone (the package isn't on PyPI yet):
git clone https://github.com/broadinstitute/carmen-analysis
cd carmen-analysis
pip install -e .

carmen-analyze \
  --assignment 1740742241_192.24_assignment.xlsx \
  --data       1740742241.csv \
  --panel      1.8_Mean \
  --output     ./out \
  [--redcap] [--barcode 1740742241]
```

A backwards-compatible legacy entry point is preserved for existing automation:

```bash
# Old behavior: globs cwd for a single .xlsx + a single .csv.
python3 analyze_run.py 1.8_Mean
python3 analyze_run.py 3_SD REDCAP
```

---

## Python API

```python
from pathlib import Path
from carmen_analysis import run_pipeline

result = run_pipeline(
    assignment_xlsx=Path("1740742241_192.24_assignment.xlsx"),
    data_csv=Path("1740742241.csv"),
    panel="1.8_Mean",
    output_dir=Path("./out"),
    redcap=False,
)

print(result.results_xlsx, result.qc_pdf, result.t13_heatmap)
```

`assignment_xlsx` and `data_csv` accept either a path-like or a binary
file-like object (used by the web app to stream uploads without disk writes).

---

## Outputs

The pipeline writes a single per-run folder:
`{output_dir}/output_{barcode}_{panel}_v{version}/`

containing three subfolders:

- **`RESULTS_{barcode}/`** — the primary deliverables.
  - `RESULTS_{barcode}_{panel}.xlsx` — 5-sheet workbook (hit calls, NTC-normalized quants, positives summary, NTC thresholds, binary results).
  - `NTC_Normalized_Heatmap_{barcode}.png` — final-timepoint heatmap.
  - `REDCAP_{barcode}_{panel}.csv` — REDCap-ready export, if `--redcap`.
- **`QUALITY_CONTROL_{barcode}/`** — QC artifacts, in two subfolders:
  - `QUALITY_CONTROL_OF_NEG_AND_POS_CONTROLS/`:
    `Quality_Control_Report_{barcode}.pdf` + `QC_{barcode}.xlsx` (or per-check CSVs as a fallback).
  - `QUALITY_CONTROL_OF_VIRAL_ASSAYS/`:
    `Assay_Performance_QC_Test_Results_{barcode}.csv` + the bundled `Assay-Level QC Test Explanation.pdf`.
- **`R&D_{barcode}/`** — intermediates for follow-up analysis (per-timepoint CSVs, per-timepoint heatmaps, normalization intermediates).

---

## Development

```bash
git clone https://github.com/broadinstitute/carmen-analysis
cd carmen-analysis
pip install -e ".[web,dev]"
pytest -q
ruff check carmen_analysis tests
```

A devcontainer is provided in `.devcontainer/`. The image and `streamlit run`
post-attach command let GitHub Codespaces serve the web app on forwarded
port 8501.

---

## Deployment

The Docker image is built and pushed by `.github/workflows/docker.yml` to
`us-central1-docker.pkg.dev/sabeti-adapt/carmen-analysis/carmen-analysis`.
Trivy scans HIGH/CRITICAL vulnerabilities and uploads SARIF results to GitHub
code scanning.

The Cloud Run deployment is fronted by an HTTPS external Application Load
Balancer with a GCP-managed SSL certificate, IAP, and a serverless NEG.
Terraform definitions live in `terraform/`.

---

## Questions?

Reach out to albeez@broadinstitute.org and cwilkason@broadinstitute.org.
