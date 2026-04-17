"""Top-level CARMEN analysis pipeline.

`run_pipeline` is the single entry point. It performs the same end-to-end
analysis as the legacy ``analyze_run.py`` script, but takes explicit input
file references and an output directory instead of globbing the current
working directory and reading from ``sys.argv``.
"""

from __future__ import annotations

import logging
import re
import shutil
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from importlib import resources
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Literal

import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from carmen_analysis import __version__ as software_version
from carmen_analysis.errors import CarmenAnalysisError
from carmen_analysis.io.matcher import DataMatcher
from carmen_analysis.io.reader import DataReader
from carmen_analysis.process.binary_results import Binary_Converter
from carmen_analysis.process.median_frame import MedianSort
from carmen_analysis.process.norm import DataProcessor
from carmen_analysis.process.ntc_con_check import ntcContaminationChecker
from carmen_analysis.process.ntcnorm import Normalized
from carmen_analysis.process.summary import Summarized
from carmen_analysis.qc.assay_qc_score import Assay_QC_Score
from carmen_analysis.qc.flags import Flagger
from carmen_analysis.qc.qual_checks import Qual_Ctrl_Checks
from carmen_analysis.qc.threshold import Thresholder
from carmen_analysis.redcap.builder import RedCapper
from carmen_analysis.viz.plotting import Plotter
from carmen_analysis.viz.t13_plotting import t13_Plotter

PanelMode = Literal["1.8_Mean", "3_SD"]
SUPPORTED_PANELS: tuple[str, ...] = ("1.8_Mean", "3_SD")

_log = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """File and folder references produced by a single pipeline run."""

    output_dir: Path
    results_dir: Path
    qc_dir: Path
    rd_dir: Path
    barcode: str
    panel: str
    software_version: str
    results_xlsx: Path
    qc_pdf: Path
    qc_xlsx: Path
    t13_heatmap: Path
    per_timepoint_heatmaps: list[Path] = field(default_factory=list)
    redcap_csv: Path | None = None


def _coerce_csv_to_bytes(data_csv: str | Path | BinaryIO | bytes) -> BytesIO:
    if isinstance(data_csv, (str, Path)):
        return BytesIO(Path(data_csv).read_bytes())
    if isinstance(data_csv, bytes):
        return BytesIO(data_csv)
    raw = data_csv.read()
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    return BytesIO(raw)


def _derive_barcode(assignment_xlsx: str | Path | BinaryIO, supplied: str | None) -> str:
    if supplied:
        return supplied
    name: str | None = None
    if isinstance(assignment_xlsx, (str, Path)):
        name = Path(assignment_xlsx).name
    else:
        name = getattr(assignment_xlsx, "name", None)
        if name:
            name = Path(name).name
    if not name:
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    match = re.match(r"^(\d+)_.*", name)
    if match:
        return match.group(1)
    return Path(name).stem


def _round_dataframe(df: pd.DataFrame, decimals: int = 5) -> pd.DataFrame:
    rounded = pd.DataFrame(index=df.index, columns=df.columns)
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            rounded.iloc[i, j] = round(df.iloc[i, j], decimals)
    return rounded


def run_pipeline(
    assignment_xlsx: str | Path | BinaryIO,
    data_csv: str | Path | BinaryIO,
    panel: PanelMode,
    output_dir: str | Path,
    *,
    redcap: bool = False,
    barcode: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> PipelineResult:
    """Run the full CARMEN analysis end-to-end.

    Parameters
    ----------
    assignment_xlsx
        Assignment-sheet XLSX. Path-like, or a file-like object opened in binary mode.
    data_csv
        Biomark instrument CSV. Path-like, or a file-like object opened in binary mode.
    panel
        Thresholding mode. ``"1.8_Mean"`` (RVP default) or ``"3_SD"`` (BBP default).
    output_dir
        Directory under which the per-run output folder is created. Created if missing.
    redcap
        If true, also produce the REDCAP-formatted export CSV.
    barcode
        Override for the IFC barcode label. By default, derived from the
        leading digits of the assignment file's name (legacy behavior).
    progress
        Optional callback invoked with progress messages. Defaults to ``logging.info``.
    """
    if panel not in SUPPORTED_PANELS:
        raise CarmenAnalysisError(
            f"Unknown panel {panel!r}. Supported: {SUPPORTED_PANELS}."
        )

    say = progress or _log.info
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    barcode_assignment = _derive_barcode(assignment_xlsx, barcode)
    say(f"IFC barcode: {barcode_assignment}")

    # ---- Output folder layout (matches legacy analyze_run.py) ----
    run_folder = output_dir / f"output_{barcode_assignment}_{panel}_v{software_version}"
    run_folder.mkdir(exist_ok=True)
    rd_subfolder = run_folder / f"R&D_{barcode_assignment}"
    res_subfolder = run_folder / f"RESULTS_{barcode_assignment}"
    qc_subfolder = run_folder / f"QUALITY_CONTROL_{barcode_assignment}"
    npc_subfolder = qc_subfolder / "QUALITY_CONTROL_OF_NEG_AND_POS_CONTROLS"
    va_subfolder = qc_subfolder / "QUALITY_CONTROL_OF_VIRAL_ASSAYS"
    for d in (rd_subfolder, res_subfolder, qc_subfolder, npc_subfolder, va_subfolder):
        d.mkdir(exist_ok=True)

    # ---- Data loading ----
    reader = DataReader()
    csv_bytes = _coerce_csv_to_bytes(data_csv)
    phrases_to_find = [
        "Raw Data for Passive Reference ROX",
        "Raw Data for Probe FAM-MGB",
        "Bkgd Data for Passive Reference ROX",
        "Bkgd Data for Probe FAM-MGB",
    ]
    csv_bytes.seek(0)
    read_dataframes, date = reader.extract_dataframes_from_csv(csv_bytes, phrases_to_find)

    # ---- Background normalization ----
    processor = DataProcessor()
    normalized_dataframes = processor.background_processing(read_dataframes)
    normalized_dataframes["signal_norm"].to_csv(rd_subfolder / "signal_norm.csv", index=True)
    normalized_dataframes["ref_norm"].to_csv(rd_subfolder / "ref_norm.csv", index=True)

    # ---- Assay/sample assignment ----
    matcher = DataMatcher()
    assigned_norms, assigned_lists = matcher.assign_assays(
        assignment_xlsx,
        normalized_dataframes["ref_norm"],
        normalized_dataframes["signal_norm"],
    )
    assigned_norms["signal_norm_raw"].to_csv(rd_subfolder / "assigned_signal_norm.csv", index=True)
    assigned_norms["ref_norm_raw"].to_csv(rd_subfolder / "assigned_ref_norm.csv", index=True)
    crRNA_assays = assigned_lists["assay_list"]
    samples_list = assigned_lists["samples_list"]

    # ---- NTC contamination check ----
    ntcCheck = ntcContaminationChecker()
    assigned_signal_norm = pd.DataFrame(assigned_norms["signal_norm_raw"]).copy()
    assigned_signal_norm_with_NTC_check = ntcCheck.ntc_cont(assigned_signal_norm)
    assigned_signal_norm_with_NTC_check.to_csv(
        rd_subfolder / "assigned_signal_norm_with_NTC_check.csv", index=True
    )

    # ---- Median frames per timepoint ----
    median = MedianSort(crRNA_assays)
    final_med_frames = median.create_median(assigned_signal_norm_with_NTC_check)
    rounded_final_med_frames = {k: _round_dataframe(df) for k, df in final_med_frames.items()}

    timepoints_subfolder = rd_subfolder / f"Quantitative Signal by Timepoint_{barcode_assignment}"
    timepoints_subfolder.mkdir(exist_ok=True)
    timepoints = list(rounded_final_med_frames.keys())
    for i, t in enumerate(timepoints, start=1):
        rounded_final_med_frames[t].to_csv(
            timepoints_subfolder / f"t{i}_{barcode_assignment}.csv", index=True
        )

    last_key = list(rounded_final_med_frames.keys())[-1]
    t13_dataframe_orig = rounded_final_med_frames[last_key]
    t13_dataframe_copy1 = pd.DataFrame(t13_dataframe_orig).copy()
    t13_dataframe_copy2 = pd.DataFrame(t13_dataframe_orig).copy()

    # ---- Thresholding ----
    thresholdr = Thresholder()
    unique_crRNA_assays_set = list(set(crRNA_assays))
    ntc_thresholds_output, t13_hit_output = thresholdr.raw_thresholder(
        unique_crRNA_assays_set,
        assigned_signal_norm_with_NTC_check,
        t13_dataframe_copy1,
        panel,
    )
    rounded_ntc_thresholds_output = _round_dataframe(ntc_thresholds_output)

    # Copies of t13_hit_output for downstream consumers
    t13_hit_output_copy1 = pd.DataFrame(t13_hit_output).copy()
    t13_hit_output_copy2 = pd.DataFrame(t13_hit_output).copy()
    t13_hit_output_copy3 = pd.DataFrame(t13_hit_output).copy()
    t13_hit_output_copy4 = pd.DataFrame(t13_hit_output).copy()

    # ---- NTC normalization ----
    ntcNorm = Normalized()
    t13_quant_norm = ntcNorm.normalizr(t13_dataframe_copy2)
    rounded_t13_quant_norm = _round_dataframe(t13_quant_norm)
    sum_row = t13_hit_output.loc[["Summary"]]
    rounded_t13_quant_norm = pd.concat((rounded_t13_quant_norm, sum_row), ignore_index=False)

    # ---- Binary conversion ----
    binary_num_converter = Binary_Converter()
    t13_hit_binary_output = binary_num_converter.hit_numeric_conv(t13_hit_output_copy4)
    t13_hit_binary_output_copy1 = pd.DataFrame(t13_hit_binary_output).copy()

    # ---- Positive-sample summary ----
    summary = Summarized()
    summary_samples_df = summary.summarizer(t13_hit_output)

    # ---- Assay-level QC score ----
    assayScorer = Assay_QC_Score()
    QC_score_per_assay_df = assayScorer.assay_level_score(t13_hit_binary_output)
    assay_lvl_QC_score_file_path = (
        va_subfolder / f"Assay_Performance_QC_Test_Results_{barcode_assignment}.csv"
    )
    QC_score_per_assay_df.to_csv(assay_lvl_QC_score_file_path, index=True)

    # Copy the bundled explanation PDF next to the QC score
    explanation_target = va_subfolder / "Assay-Level QC Test Explanation.pdf"
    with resources.as_file(
        resources.files("carmen_analysis.resources").joinpath(
            "Assay-Level QC Test Explanation.pdf"
        )
    ) as explanation_src:
        shutil.copy(explanation_src, explanation_target)

    say(
        "The four quality control tests to evaluate assay performance are complete. "
        f"Their results have been saved to {va_subfolder}"
    )

    # ---- Quality-control checks (PDF + Excel) ----
    qual_checks = Qual_Ctrl_Checks()
    npc_pdf_file_path = npc_subfolder / f"Quality_Control_Report_{barcode_assignment}.pdf"
    qc_output_file_path = npc_subfolder / f"QC_{barcode_assignment}.xlsx"

    rnasep_df = pd.DataFrame()
    high_raw_ntc_signal_df = pd.DataFrame()
    fail_nocrRNA_check_df = pd.DataFrame()

    rnasep_df, high_raw_ntc_signal_df, fail_nocrRNA_check_df = _write_qc_artifacts(
        qual_checks=qual_checks,
        npc_subfolder=npc_subfolder,
        npc_pdf_file_path=npc_pdf_file_path,
        qc_output_file_path=qc_output_file_path,
        barcode_assignment=barcode_assignment,
        t13_hit_output_copy1=t13_hit_output_copy1,
        t13_hit_output_copy2=t13_hit_output_copy2,
        t13_hit_output_copy3=t13_hit_output_copy3,
        assigned_norms=assigned_norms,
        t13_hit_binary_output_copy1=t13_hit_binary_output_copy1,
        say=say,
    )

    # ---- Apply flags ----
    flagger = Flagger()
    invalid_assays, invalid_samples, flagged_files, _processed_samples = flagger.assign_flags(
        fail_nocrRNA_check_df,
        high_raw_ntc_signal_df,
        rnasep_df,
        QC_score_per_assay_df,
        t13_hit_output,
        rounded_t13_quant_norm,
        summary_samples_df,
        rounded_ntc_thresholds_output,
        t13_hit_binary_output,
    )
    fl_t13_hit_output = flagged_files[0]
    fl_rounded_t13_quant_norm = flagged_files[1]
    fl_summary_samples_df = flagged_files[2]
    fl_rounded_ntc_thresholds_output = flagged_files[3]
    fl_t13_hit_binary_output = flagged_files[4]
    fl_t13_hit_binary_output.columns = fl_t13_hit_binary_output.columns.str.upper()

    # ---- Write the unified results workbook ----
    sheet_names = [
        "CARMEN_Hit_Results",
        "NTC_Normalized_Quant_Results",
        "Summary_of_Positive_Samples",
        "NTC_thresholds",
        "Binary_Results",
    ]
    dataframes = [
        fl_t13_hit_output,
        fl_rounded_t13_quant_norm,
        fl_summary_samples_df,
        fl_rounded_ntc_thresholds_output,
        fl_t13_hit_binary_output,
    ]
    output_file_path = res_subfolder / f"RESULTS_{barcode_assignment}_{panel}.xlsx"
    _write_results_workbook(output_file_path, dataframes, sheet_names, say=say)
    say(f"All FLAGGED FILES have been saved to {output_file_path}.")

    # ---- Per-timepoint heatmaps ----
    heatmap_generator = Plotter()
    tgap = 3
    unique_crRNA_assays = list(OrderedDict.fromkeys(crRNA_assays))
    heatmap = heatmap_generator.plt_heatmap(
        tgap, barcode_assignment, final_med_frames, samples_list, unique_crRNA_assays, timepoints
    )
    heatmaps_subfolder = rd_subfolder / f"Heatmaps_by_Timepoint_{barcode_assignment}"
    heatmaps_subfolder.mkdir(exist_ok=True)
    per_timepoint_paths: list[Path] = []
    for i, t in enumerate(timepoints, start=1):
        heatmap_filename = heatmaps_subfolder / f"Heatmap_t{i}_{barcode_assignment}.png"
        heatmap[t].savefig(heatmap_filename, bbox_inches="tight", dpi=80)
        plt.close(heatmap[t])
        per_timepoint_paths.append(heatmap_filename)
    say(f"Per-timepoint heatmaps saved to {heatmaps_subfolder}")

    # ---- Final t13 NTC-normalized heatmap ----
    t13_heatmap_generator = t13_Plotter()
    heatmap_t13_quant_norm = t13_heatmap_generator.t13_plt_heatmap(
        tgap,
        barcode_assignment,
        t13_quant_norm,
        samples_list,
        unique_crRNA_assays,
        timepoints,
        invalid_samples,
        invalid_assays,
        rnasep_df,
    )
    heatmap_t13_quant_norm_filename = (
        res_subfolder / f"NTC_Normalized_Heatmap_{barcode_assignment}.png"
    )
    heatmap_t13_quant_norm.savefig(
        heatmap_t13_quant_norm_filename, bbox_inches="tight", dpi=80
    )
    plt.close(heatmap_t13_quant_norm)

    # ---- Optional REDCap export ----
    redcap_path: Path | None = None
    if redcap:
        redcapper = RedCapper()
        fl_t13_hit_binary_output_2 = fl_t13_hit_binary_output.copy()
        redcap_t13_hit_binary_output, _samplesDF = redcapper.build_redcap(
            fl_t13_hit_binary_output_2, date, barcode_assignment, panel, software_version
        )
        redcap_path = res_subfolder / f"REDCAP_{barcode_assignment}_{panel}.csv"
        redcap_t13_hit_binary_output.to_csv(redcap_path, index=False)
        say("REDCAP file generated.")

    say("Operation complete.")

    return PipelineResult(
        output_dir=run_folder,
        results_dir=res_subfolder,
        qc_dir=qc_subfolder,
        rd_dir=rd_subfolder,
        barcode=barcode_assignment,
        panel=panel,
        software_version=software_version,
        results_xlsx=output_file_path,
        qc_pdf=npc_pdf_file_path,
        qc_xlsx=qc_output_file_path,
        t13_heatmap=heatmap_t13_quant_norm_filename,
        per_timepoint_heatmaps=per_timepoint_paths,
        redcap_csv=redcap_path,
    )


def _write_qc_artifacts(
    *,
    qual_checks: Qual_Ctrl_Checks,
    npc_subfolder: Path,
    npc_pdf_file_path: Path,
    qc_output_file_path: Path,
    barcode_assignment: str,
    t13_hit_output_copy1: pd.DataFrame,
    t13_hit_output_copy2: pd.DataFrame,
    t13_hit_output_copy3: pd.DataFrame,
    assigned_norms: dict,
    t13_hit_binary_output_copy1: pd.DataFrame,
    say: Callable[[str], None],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the consolidated QC PDF and Excel workbook.

    Returns ``(rnasep_df, high_raw_ntc_signal_df, fail_nocrRNA_check_df)`` for
    downstream flagging, regardless of whether the workbook write succeeded.
    """
    doc = SimpleDocTemplate(
        str(npc_pdf_file_path),
        pagesize=A4,
        leftMargin=50,
        rightMargin=50,
        topMargin=50,
        bottomMargin=50,
    )
    styles = getSampleStyleSheet()
    style = styles["Normal"]
    header_style = styles["Heading2"]
    header_style.fontName = "Times-Roman"
    header_style.fontSize = 12
    header_style.textColor = (0, 0, 0.5)
    style.fontName = "Times-Roman"
    style.fontSize = 12

    content: list = []

    QC_sheet_names = [
        "Positive NDC Samples",
        "Negative CPC Samples",
        "Negative RNaseP Samples",
        "Positive NTC Samples",
        "Potentially Co-infected Samples",
        "Invalid Samples",
    ]

    ndc_text_long = (
        "If any of the NDC samples show a positive result for any assay, then that assay should "
        "be evaluated for contamination with nucleases likely at the sample mastermix preparation "
        "step in the experimental workflow. However, other sources for NDC contamination may exist.\n"
    )
    rnasep_long = [
        "Possible reasons for a sample testing negative for the RNaseP assay:",
        "(A) If the sample is negative for all assays (including RNaseP), then the most plausible hypothesis is that the viral extraction protocol used in this experiment needs to be examined. For optimal results, the extraction must be compatible with the Standard Operating Procedure (SOP) advised by the CARMEN team in the Sabeti Lab.",
        "** Note: If the sample is negative for RNaseP and ALL other crRNA assays tested in this experiment, the sample should be rendered invalid.",
        "(B) If the sample is negative for RNaseP BUT positive for any other viral crRNA assay (excluding RNaseP or no-crRNA), then the most plausible hypothesis is that the sample\u2019s viral titer may be too high compared to its RNaseP titer. This, thereby, renders the system possibly unable to detect RNaseP, leading to the sample testing negative for RNaseP.",
        "** Note: If the sample is negative for RNaseP but positive for any other viral crRNA assay (excluding RNaseP or no-crRNA) tested in this experiment, the sample can still be included in the final results.",
        "(C) The source sample may have insufficient material, leading to a negative RNaseP signal and an invalid sample result.",
        "Please be advised to check the output files as well.",
    ]
    coinf_paragraphs = [
        "A preliminary evaluation for co-infection of a given sample against all tested assays has been completed:",
        "   (A) If you have included Combined Positive Controls (CPCs) in this experiment, as recommended, these positive controls should be identified and listed among the flagged samples. CPCs are expected to show a \u201cco-detection\u201d with ALL of the assays being tested in this experiment.",
        "   (B) Samples are not flagged as \u201cco-detected\u201d based on positivity with RNaseP and a second assay. For a sample to be flagged during this Co-detection Check, it must test positive for at least two assays, excluding RNaseP.",
        "   (C) All other flagged samples should be further evaluated for potential co-infection.",
        "Please be advised to check the output files to further evaluate potential co-infection.",
    ]

    # First pass: try the unified Excel workbook
    try:
        with pd.ExcelWriter(qc_output_file_path, engine="xlsxwriter") as writer:
            # (1) NDC
            ndc_positives_df = qual_checks.ndc_check(t13_hit_output_copy1)
            _write_qc_sheet(writer, ndc_positives_df, QC_sheet_names[0], qc_output_file_path, say)
            content.append(Paragraph("1. Evaluation of No Detect Control (NDC) Contamination", header_style))
            content.append(Spacer(1, 0.2 * inch))
            ndc_text = (
                [
                    f"Please consult NDC_Check_{barcode_assignment}.csv to see the initial evaluation of the NDC negative controls tested in this experiment. In this file, assays are flagged for which the NDC samples have tested positive, after being thresholded against the assay-specific NTC mean.",
                    ndc_text_long,
                    "Please be advised to check the output files as well.",
                ]
                if not ndc_positives_df.empty
                else [
                    "Since none of the NDCs ran in this experiment appear positive, there is likely no NDC contamination.",
                    "Please check the output files as well.",
                ]
            )
            _append_paragraphs(content, ndc_text, style)

            # (2) CPC
            cpc_negatives_df = qual_checks.cpc_check(t13_hit_output_copy2)
            _write_qc_sheet(writer, cpc_negatives_df, QC_sheet_names[1], qc_output_file_path, say)
            content.append(Paragraph("2. Evaluation of Combined Positive Control (CPC) Validity", header_style))
            content.append(Spacer(1, 0.2 * inch))
            cpc_text = (
                [
                    f"Please consult CPC_Check_{barcode_assignment}.csv to see the initial evaluation of the CPC positive controls tested in this experiment. In this file, assays are flagged for which the CPC samples have tested negative, after being thresholded against the assay-specific NTC mean.",
                    "If any of the CPC samples show a negative result for any assay excluding the 'no-crRNA' negative control assay, then that assay should be considered invalid for this experiment.",
                    "Please be advised to check the output files as well.",
                ]
                if not cpc_negatives_df.empty
                else [
                    "Warning: First verify that your experiment included a CPC sample. If yes, proceed to the following CPC analysis.",
                    "After thresholding against the NTC, the CPC(s) appears as positive for all crRNA assays tested. However, it is expected for the CPC(s) to test as negative for 'no-crRNA' assay. There may be possible contamination of the 'no-crRNA' assay.",
                    "Please be advised to check the output files as well.",
                ]
            )
            _append_paragraphs(content, cpc_text, style)

            # (3) RNaseP
            rnasep_df = qual_checks.rnasep_check(t13_hit_output_copy3)
            _write_qc_sheet(writer, rnasep_df, QC_sheet_names[2], qc_output_file_path, say)
            content.append(
                Paragraph("3. Evaluation of Human Samples for the Internal Control (RNaseP)", header_style)
            )
            content.append(Spacer(1, 0.2 * inch))
            rnasep_text = (
                [
                    "Warning: First verify that your experiment included a RNaseP assay. If yes, proceed to the following RNaseP analysis.",
                    f"Please consult RNaseP_Check_{barcode_assignment}.csv to see which samples are negative for the RNaseP assay(s). In this file, the samples that appear negative for the RNaseP assays have been flagged after thresholding against the NTC. The negative controls (NTC and NDC) are expected to be negative for the RNaseP assay and should be listed here (if you have included them in this experiment). All other samples should be evaluated for being negative for the RNaseP assay.",
                    *rnasep_long,
                ]
                if not rnasep_df.empty
                else [
                    "Warning: First verify that your experiment included a RNaseP assay. If yes, proceed to the following RNaseP analysis.",
                    "All samples (including negative controls) have tested positive for the RNaseP assay(s) tested in this experiment. However, the assay(s) for RNaseP internal control should test negative for the NTC and NDC negative control.",
                    "There are a few different reasons that all samples test positive for RNaseP. The most plausible hypothesis is that there is RNaseP contamination in this experiment. Precaution is advised to mitigate contamination avenues, especially at the RT-PCR (nucleic acid amplification) stage.",
                    "Please be advised to check the output files as well.",
                ]
            )
            _append_paragraphs(content, rnasep_text, style)

            # (4) NTC
            assigned_signal_norm_2 = pd.DataFrame(assigned_norms["signal_norm_raw"]).copy()
            high_raw_ntc_signal_df = qual_checks.ntc_check(assigned_signal_norm_2)
            _write_qc_sheet(writer, high_raw_ntc_signal_df, QC_sheet_names[3], qc_output_file_path, say)
            content.append(Paragraph("4. Evaluation of No Target Control (NTC) Contamination", header_style))
            content.append(Spacer(1, 0.2 * inch))
            ntc_text = (
                [
                    f"Please consult NTC_Contamination_Check_{barcode_assignment}.csv to see which NTC samples may be potentially contaminated.",
                    "This file contains a list of samples that have a raw fluorescence signal above 0.5 a.u. These samples are being flagged for having a higher than normal signal for an NTC sample. The range for typical raw fluorescence signal for an NTC sample is between 0.1 and 0.5 a.u.",
                    "Please be advised to check the output files to further evaluate potential NTC contamination.",
                ]
                if not high_raw_ntc_signal_df.empty
                else [
                    "The raw fluorescence signal for each NTC sample across all crRNA assays tested in this experiment appears to be within the normal range of 0.1 and 0.5 a.u. Risk of NTC contamination is low.",
                    "Please be advised to check the output files as well.",
                ]
            )
            _append_paragraphs(content, ntc_text, style)

            # (5) Co-infection
            coinfection_df = qual_checks.coinf_check(t13_hit_binary_output_copy1)
            _write_qc_sheet(writer, coinfection_df, QC_sheet_names[4], qc_output_file_path, say)
            content.append(Paragraph("5. Evaluation of Potential Co-Infected Samples", header_style))
            content.append(Spacer(1, 0.2 * inch))
            coinf_intro = f"Please consult Codetection_Check_{barcode_assignment}.csv to see which samples may be potentially co-infected."
            _append_paragraphs(content, [coinf_intro, *coinf_paragraphs], style)

            doc.build(content)

            # (6) No-crRNA
            fail_nocrRNA_check_df = qual_checks.no_crrna_check(t13_hit_binary_output_copy1)
            _write_qc_sheet(writer, fail_nocrRNA_check_df, QC_sheet_names[5], qc_output_file_path, say)

    except Exception as exc:  # noqa: BLE001 — preserve legacy fallback
        say(
            f"Error with QC Excel file generation: {exc}. Results are saved as individual CSV files."
        )
        ndc_positives_df = qual_checks.ndc_check(t13_hit_output_copy1)
        _write_qc_csv(npc_subfolder, f"NDC_Check_{barcode_assignment}.csv", ndc_positives_df, say,
                      empty_msg="For all viral assays tested in this experiment, all NDC samples test negative.")
        cpc_negatives_df = qual_checks.cpc_check(t13_hit_output_copy2)
        _write_qc_csv(npc_subfolder, f"CPC_Check_{barcode_assignment}.csv", cpc_negatives_df, say,
                      empty_msg="For all viral assays tested in this experiment, all CPC samples test positive.")
        rnasep_df = qual_checks.rnasep_check(t13_hit_output_copy3)
        _write_qc_csv(npc_subfolder, f"RNaseP_Check_{barcode_assignment}.csv", rnasep_df, say,
                      empty_msg="Please verify that you have included an RNaseP viral assay in your CARMEN experiment. If you have, all samples test postive for the RNaseP assay.")
        assigned_signal_norm_2 = pd.DataFrame(assigned_norms["signal_norm_raw"]).copy()
        high_raw_ntc_signal_df = qual_checks.ntc_check(assigned_signal_norm_2)
        _write_qc_csv(npc_subfolder, f"NTC_Contamination_Check_{barcode_assignment}.csv",
                      high_raw_ntc_signal_df, say,
                      empty_msg="For all viral assays tested in this experiment, there are no NTC samples which appear as contaminated.")
        coinfection_df = qual_checks.coinf_check(t13_hit_binary_output_copy1)
        _write_qc_csv(npc_subfolder, f"Coinfection_Check_{barcode_assignment}.csv", coinfection_df, say,
                      empty_msg="For all viral assays tested in this experiment, there are no samples which appear as potentially co-infected.")
        fail_nocrRNA_check_df = qual_checks.no_crrna_check(t13_hit_binary_output_copy1)
        _write_qc_csv(npc_subfolder, f"No_crRNA_Assay_Check_{barcode_assignment}.csv",
                      fail_nocrRNA_check_df, say,
                      empty_msg="All samples have tested against the no-crRNA assay. Thus, there are no samples which must be invalidated.")
        try:
            doc.build(content)
        except Exception:
            pass

    return rnasep_df, high_raw_ntc_signal_df, fail_nocrRNA_check_df


def _write_qc_sheet(
    writer: pd.ExcelWriter,
    df: pd.DataFrame,
    sheet_name: str,
    qc_output_file_path: Path,
    say: Callable[[str], None],
) -> None:
    if df.empty:
        empty_message_df = pd.DataFrame({"Message": [_qc_empty_message(sheet_name)]})
        empty_message_df.to_excel(writer, sheet_name=sheet_name, index=False)
        say(f"File generated with message in {sheet_name} at {qc_output_file_path}")
    else:
        df.to_excel(writer, sheet_name=sheet_name, index=True)
        say(f"File generated with data in {sheet_name} at {qc_output_file_path}")


def _qc_empty_message(sheet_name: str) -> str:
    msgs = {
        "Positive NDC Samples": "For all viral assays tested in this experiment, all NDC samples test negative.",
        "Negative CPC Samples": "For all viral assays tested in this experiment, all CPC samples test positive.",
        "Negative RNaseP Samples": "Please verify that you have included an RNaseP viral assay in your CARMEN experiment. If you have, all samples test postive for the RNaseP assay.",
        "Positive NTC Samples": "For all viral assays tested in this experiment, there are no NTC samples which appear as contaminated.",
        "Potentially Co-infected Samples": "For all viral assays tested in this experiment, there are no samples which appear as potentially co-infected.",
        "Invalid Samples": "All samples have tested against the no-crRNA assay. Thus, there are no samples which must be invalidated.",
    }
    return msgs.get(sheet_name, "No data.")


def _write_qc_csv(
    npc_subfolder: Path,
    filename: str,
    df: pd.DataFrame,
    say: Callable[[str], None],
    *,
    empty_msg: str,
) -> None:
    target = npc_subfolder / filename
    if df.empty:
        pd.DataFrame({"Message": [empty_msg]}).to_csv(target, index=False, header=False)
        say(f"CSV created with message at {target}")
    else:
        df.to_csv(target, index=True)
        say(f"CSV created with data at {target}")


def _append_paragraphs(content: list, lines: list[str], style) -> None:
    for line in lines:
        content.append(Paragraph(line, style))
        content.append(Spacer(1, 0.1 * inch))


def _write_results_workbook(
    output_file_path: Path,
    dataframes: list[pd.DataFrame],
    sheet_names: list[str],
    *,
    say: Callable[[str], None],
) -> None:
    try:
        with pd.ExcelWriter(output_file_path, engine="xlsxwriter") as writer:
            red_cells: set[tuple[int, int]] = set()
            red_font = writer.book.add_format({"font_color": "F97609", "bold": True})
            green_font = writer.book.add_format({"font_color": "1A85FF"})
            black_font = writer.book.add_format({"font_color": "000000"})

            for df, sheet_name in zip(dataframes, sheet_names, strict=True):
                df.fillna("", inplace=True)
                df.to_excel(writer, sheet_name=sheet_name, index=True)
                worksheet = writer.sheets[sheet_name]
                worksheet.set_column(0, 0, 20)
                for col_idx in range(1, df.shape[1]):
                    worksheet.set_column(col_idx, col_idx, 16)

                if sheet_name == "CARMEN_Hit_Results":
                    for row_idx, row in enumerate(df.values, start=1):
                        for col_idx, cell_value in enumerate(row, start=1):
                            cell_str = str(cell_value)
                            if "NEGATIVE" in cell_str:
                                worksheet.write(row_idx, col_idx, cell_value, green_font)
                            elif "POSITIVE" in cell_str:
                                worksheet.write(row_idx, col_idx, cell_value, red_font)
                                red_cells.add((row_idx, col_idx))
                            else:
                                worksheet.write(row_idx, col_idx, cell_value, black_font)
                if sheet_name in {"NTC_Normalized_Quant_Results", "Binary_Results"}:
                    for row_idx, row in enumerate(df.values, start=1):
                        for col_idx, cell_value in enumerate(row, start=1):
                            if (row_idx, col_idx) in red_cells:
                                worksheet.write(row_idx, col_idx, cell_value, red_font)
    except Exception as exc:  # noqa: BLE001 — preserve legacy fallback
        say(f"Error saving single Excel file: {exc}. Results are saved as 5 individual csv files.")
        # Best-effort fallback writes (CSV) — the same pattern the legacy script used.
        for df, sheet_name in zip(dataframes, sheet_names, strict=True):
            df.to_csv(output_file_path.with_name(f"{sheet_name}.csv"), index=True)
