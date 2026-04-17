"""Stateless Streamlit web UI for the CARMEN analysis pipeline.

Usage:
    streamlit run -m carmen_analysis.web.app
    # or, after `pip install carmen-analysis[web]`:
    python -m streamlit run $(python -c "import carmen_analysis.web.app as a; print(a.__file__)")

The app does not retain user data between sessions. Each run writes outputs to
a fresh temporary directory which is zipped, offered for download, and then
discarded when the Streamlit session ends.
"""

from __future__ import annotations

import io
import shutil
import tempfile
import zipfile
from pathlib import Path

import streamlit as st

from carmen_analysis import __version__
from carmen_analysis.errors import CarmenAnalysisError
from carmen_analysis.pipeline import SUPPORTED_PANELS, run_pipeline


def _zip_directory(source: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in source.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(source.parent))
    return buf.getvalue()


def render() -> None:
    st.set_page_config(page_title="CARMEN Analysis", layout="centered")
    st.title("CARMEN Analysis")
    st.caption(f"Pipeline version {__version__}")

    st.markdown(
        "Upload your **assignment sheet** (XLSX) and **Biomark instrument export** (CSV), "
        "choose a thresholding mode, then click **Run analysis**. Outputs are bundled "
        "into a single ZIP download. Nothing is retained on the server after your session ends."
    )

    with st.form("carmen-run", clear_on_submit=False):
        assignment_upload = st.file_uploader(
            "Assignment sheet (.xlsx)", type=["xlsx"], accept_multiple_files=False
        )
        data_upload = st.file_uploader(
            "Biomark instrument export (.csv)", type=["csv"], accept_multiple_files=False
        )
        panel = st.selectbox(
            "Thresholding mode",
            options=SUPPORTED_PANELS,
            help="'1.8_Mean' is the default for RVP runs. '3_SD' is the default for BBP runs.",
        )
        redcap = st.checkbox(
            "Also generate REDCAP-formatted CSV", value=False,
            help="Produces an additional CSV ready for upload to REDCap.",
        )
        barcode_override = st.text_input(
            "IFC barcode (optional)", value="",
            help="Leave empty to derive from the assignment file name.",
        )
        submitted = st.form_submit_button("Run analysis", type="primary")

    if not submitted:
        return

    if assignment_upload is None or data_upload is None:
        st.error("Please upload both the assignment XLSX and the instrument CSV.")
        return

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

        with st.spinner("Running analysis..."):
            result = run_pipeline(
                assignment_xlsx=assignment_path,
                data_csv=data_path,
                panel=panel,
                output_dir=workdir / "out",
                redcap=redcap,
                barcode=barcode_override.strip() or None,
                progress=progress,
            )

        st.success(f"Analysis complete: barcode {result.barcode}, panel {result.panel}.")

        zip_bytes = _zip_directory(result.output_dir)
        st.download_button(
            label="Download all outputs (.zip)",
            data=zip_bytes,
            file_name=f"{result.output_dir.name}.zip",
            mime="application/zip",
        )

        results_xlsx = result.results_xlsx
        if results_xlsx.exists():
            st.download_button(
                label=f"Download {results_xlsx.name}",
                data=results_xlsx.read_bytes(),
                file_name=results_xlsx.name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        if result.t13_heatmap.exists():
            st.image(str(result.t13_heatmap), caption="NTC-normalized t13 heatmap")
    except CarmenAnalysisError as exc:
        st.error(f"Pipeline error: {exc}")
    except Exception as exc:  # noqa: BLE001 — surface unexpected errors to the user
        st.exception(exc)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    render()
else:
    render()
