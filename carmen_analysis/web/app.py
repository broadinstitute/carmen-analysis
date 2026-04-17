"""Stateless Streamlit web UI for the CARMEN analysis pipeline.

Usage (any of these):

    carmen-web                                              # console script
    streamlit run path/to/carmen_analysis/web/app.py
    python -m streamlit run "$(python -c 'import carmen_analysis.web.app as a; print(a.__file__)')"

The app does not retain user data between sessions. Each run writes outputs to
a fresh temporary directory which is zipped, offered for download, and then
discarded when the Streamlit session ends.
"""

from __future__ import annotations

import io
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from carmen_analysis import __version__
from carmen_analysis.errors import CarmenAnalysisError
from carmen_analysis.pipeline import PipelineResult, run_pipeline

# ----- Panel options shown in the UI ---------------------------------------- #

@dataclass(frozen=True)
class _PanelChoice:
    label: str
    panels: tuple[str, ...]   # one panel = run once; two = "run both and compare"


_PANEL_CHOICES: tuple[_PanelChoice, ...] = (
    _PanelChoice("Respiratory Virus Panel — RVP (1.8 × NTC mean)", ("1.8_Mean",)),
    _PanelChoice("Blood-Borne Pathogens — BBP (3 × NTC standard deviation)", ("3_SD",)),
    _PanelChoice("Run both and compare", ("1.8_Mean", "3_SD")),
)


# ----- Helpers -------------------------------------------------------------- #

def _zip_directory(source: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in source.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(source.parent))
    return buf.getvalue()


def _binary_results_dataframe(result: PipelineResult) -> pd.DataFrame | None:
    """Return the Binary_Results sheet from the results workbook, if present.

    Surfaced separately so Kyle's "find/replace negative→0, positive→1" step
    becomes a one-click download instead of manual work.
    """
    try:
        return pd.read_excel(result.results_xlsx, sheet_name="Binary_Results", index_col=0)
    except Exception:
        return None


def _positives_summary_dataframe(result: PipelineResult) -> pd.DataFrame | None:
    """Return the per-assay positive-count summary row from the results workbook."""
    try:
        df = pd.read_excel(result.results_xlsx, sheet_name="CARMEN_Hit_Results", index_col=0)
    except Exception:
        return None
    if "Summary" not in df.index:
        return None
    summary = df.loc[["Summary"]].T
    summary.columns = ["Positive samples"]
    summary.index.name = "Assay"
    return summary


def _df_to_tsv_bytes(df: pd.DataFrame) -> bytes:
    """Tab-separated, no index, suitable for paste into GraphPad Prism."""
    return df.to_csv(sep="\t", index=True).encode("utf-8")


# ----- Result rendering ----------------------------------------------------- #

def _render_result(result: PipelineResult, label_suffix: str = "") -> None:
    import streamlit as st

    suffix = f" ({label_suffix})" if label_suffix else ""
    st.success(f"Analysis complete{suffix}: barcode {result.barcode}, panel {result.panel}.")

    # Headline: per-assay positive-sample counts.
    positives = _positives_summary_dataframe(result)
    if positives is not None:
        st.subheader(f"Positive samples per assay{suffix}")
        st.dataframe(positives, use_container_width=True)

    # Surface the binary CSV prominently — it's what wet-lab users paste into
    # GraphPad Prism. Without this they manually find/replace POSITIVE→1.
    binary = _binary_results_dataframe(result)
    if binary is not None:
        st.subheader(f"Binary results (1 = POSITIVE, 0 = NEGATIVE){suffix}")
        st.dataframe(binary, use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label=f"Download binary results as TSV (Prism-ready){suffix}",
                data=_df_to_tsv_bytes(binary),
                file_name=f"binary_{result.barcode}_{result.panel}.tsv",
                mime="text/tab-separated-values",
            )
        with col2:
            st.download_button(
                label=f"Download binary results as CSV{suffix}",
                data=binary.to_csv(index=True).encode("utf-8"),
                file_name=f"binary_{result.barcode}_{result.panel}.csv",
                mime="text/csv",
            )

    # Final t13 heatmap inline.
    if result.t13_heatmap.exists():
        st.subheader(f"NTC-normalized heatmap (final timepoint){suffix}")
        st.image(str(result.t13_heatmap))

    # Per-timepoint heatmaps in a collapsed expander.
    if result.per_timepoint_heatmaps:
        with st.expander(f"Per-timepoint heatmaps{suffix} ({len(result.per_timepoint_heatmaps)} files)"):
            for hm in result.per_timepoint_heatmaps:
                st.image(str(hm), caption=hm.name)

    # Always offer the full results workbook + the QC PDF.
    cols = st.columns(3)
    with cols[0]:
        if result.results_xlsx.exists():
            st.download_button(
                label=f"Results workbook (.xlsx){suffix}",
                data=result.results_xlsx.read_bytes(),
                file_name=result.results_xlsx.name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    with cols[1]:
        if result.qc_pdf.exists():
            st.download_button(
                label=f"QC report (.pdf){suffix}",
                data=result.qc_pdf.read_bytes(),
                file_name=result.qc_pdf.name,
                mime="application/pdf",
            )
    with cols[2]:
        if result.redcap_csv and result.redcap_csv.exists():
            st.download_button(
                label=f"REDCap export (.csv){suffix}",
                data=result.redcap_csv.read_bytes(),
                file_name=result.redcap_csv.name,
                mime="text/csv",
            )


# ----- Main render flow ----------------------------------------------------- #

def render() -> None:
    import streamlit as st

    st.set_page_config(page_title="CARMEN Analysis", layout="centered")
    st.title("CARMEN Analysis")
    st.caption(f"Pipeline version {__version__}")

    st.markdown(
        "Upload your **assignment sheet** (XLSX) and **Biomark instrument export** (CSV), "
        "choose your panel, then click **Analyze**. Outputs are bundled into a single ZIP "
        "download. Nothing is retained on the server after your session ends."
    )

    with st.form("carmen-run", clear_on_submit=False):
        assignment_upload = st.file_uploader(
            "Assignment sheet (.xlsx)", type=["xlsx"], accept_multiple_files=False
        )
        data_upload = st.file_uploader(
            "Biomark instrument export (.csv)", type=["csv"], accept_multiple_files=False
        )
        panel_label = st.radio(
            "Panel / thresholding mode",
            options=[choice.label for choice in _PANEL_CHOICES],
            index=0,
            help=(
                "RVP uses 1.8 × the mean of the NTC controls per assay. "
                "BBP uses 3 standard deviations above the NTC mean per assay. "
                "'Run both and compare' produces side-by-side results for the same dataset."
            ),
        )
        redcap = st.checkbox(
            "Also generate REDCap-formatted CSV", value=False,
            help="Produces an additional CSV ready for upload to REDCap.",
        )
        barcode_override = st.text_input(
            "IFC barcode (optional)", value="",
            help="Leave empty to derive from the assignment file name.",
        )
        submitted = st.form_submit_button("Analyze", type="primary")

    if not submitted:
        return

    if assignment_upload is None or data_upload is None:
        st.error("Please upload both the assignment XLSX and the instrument CSV.")
        return

    panel_choice = next(c for c in _PANEL_CHOICES if c.label == panel_label)

    progress_area = st.empty()
    log_lines: list[str] = []

    def progress(msg: str) -> None:
        log_lines.append(msg)
        progress_area.code("\n".join(log_lines), language="text")

    workdir = Path(tempfile.mkdtemp(prefix="carmen-"))
    try:
        assignment_path = workdir / assignment_upload.name
        assignment_path.write_bytes(assignment_upload.getvalue())
        data_path = workdir / data_upload.name
        data_path.write_bytes(data_upload.getvalue())

        results: list[PipelineResult] = []
        for panel in panel_choice.panels:
            with st.spinner(f"Running analysis ({panel})..."):
                result = run_pipeline(
                    assignment_xlsx=assignment_path,
                    data_csv=data_path,
                    panel=panel,
                    output_dir=workdir / f"out-{panel}",
                    redcap=redcap,
                    barcode=barcode_override.strip() or None,
                    progress=progress,
                )
            results.append(result)

        if len(results) == 1:
            _render_result(results[0])
            zip_bytes = _zip_directory(results[0].output_dir)
            st.download_button(
                label="Download all outputs (.zip)",
                data=zip_bytes,
                file_name=f"{results[0].output_dir.name}.zip",
                mime="application/zip",
            )
        else:
            tabs = st.tabs([f"Results — {r.panel}" for r in results])
            for tab, r in zip(tabs, results, strict=True):
                with tab:
                    _render_result(r, label_suffix=r.panel)
            # Combined zip across both runs.
            combined = workdir / f"carmen_{results[0].barcode}_both"
            combined.mkdir(exist_ok=True)
            for r in results:
                shutil.copytree(r.output_dir, combined / r.output_dir.name, dirs_exist_ok=True)
            st.download_button(
                label="Download both runs (.zip)",
                data=_zip_directory(combined),
                file_name=f"{combined.name}.zip",
                mime="application/zip",
            )
    except CarmenAnalysisError as exc:
        st.error(f"Pipeline error: {exc}")
    except Exception as exc:  # noqa: BLE001 — surface unexpected errors to the user
        st.exception(exc)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def run() -> None:
    """Console-script entry point: launches this module under Streamlit.

    Wired up as ``carmen-web`` in ``pyproject.toml`` so non-technical users
    don't have to remember the streamlit invocation.
    """
    import sys

    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", __file__, *sys.argv[1:]]
    sys.exit(stcli.main())


def _is_streamlit_runtime() -> bool:
    """True when this module is being executed by `streamlit run`."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


# Streamlit executes the script as `__main__`; safe-guard so plain
# `import carmen_analysis.web.app` (e.g. for smoke tests) does not render.
if __name__ == "__main__" or _is_streamlit_runtime():
    render()
