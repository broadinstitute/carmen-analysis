#dataloading assets
import sys
import pandas as pd
import numpy as np 
import math 
#plotting imports
import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns 
#file imports
from io import BytesIO
from datetime import datetime
import base64
from pathlib import Path
import os 
from os import path
import shutil
import glob 
import re
from collections import OrderedDict
from reader import DataReader
from norm import DataProcessor
from matcher import DataMatcher
from median_frame import MedianSort
from threshold import Thresholder
from openpyxl import Workbook
from openpyxl.styles import Font
from ntcnorm import Normalized
from summary import Summarized
from plotting import Plotter
from t13_plotting import t13_Plotter
import matplotlib.patches as patches
from tqdm import tqdm
# quality control checks imports
from qual_checks import Qual_Ctrl_Checks
from binary_results import Binary_Converter
from ntc_con_check import ntcContaminationChecker
from assay_qc_score import Assay_QC_Score
import csv
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
# annotations and flags imports
from flags import Flagger
import xlsxwriter
# generate redcap
from redcap_builder import RedCapper



######################################################################################################################################################
# assign software version
software_version = '5.4.2'

######################################################################################################################################################
# data loading
all_files = list(Path(os.getcwd()).glob('*'))

assignment_files = sorted([fname for fname in all_files if fname.suffix == ".xlsx"])

# first load in the assignment file 
if len(assignment_files) == 0:
    print("Please upload an Assignment Sheet.")
    sys.exit(1)
elif len(assignment_files) > 1:
    print(f"Multiple Assignment Sheets found in the folder: {assignment_files}")
else:
    assignment_file = assignment_files[0]
    assignment_file_name = assignment_file.name
    print(f"This is the Assignment Sheet that was loaded: {assignment_file_name}")
    match = re.match(r"^(\d+)_.*", assignment_file_name)
    barcode_assignment = ""
    if match:
        s= match.group(1)
        barcode_assignment = match.group(1)
    print(f"IFC barcode: {barcode_assignment}")

######################################################################################################################################################
# instantiate Data Reader from reader.py
reader = DataReader() # make am empty class object that corresponds to the DataReader() object from reader.py

# then load in the data file 
data_files = sorted([fname for fname in all_files if fname.suffix == ".csv"])
if len(data_files) == 0:
    print("Please upload a Data File.")
    sys.exit(1)
elif len(data_files) > 1:
    print(f"Multiple data files were uploaded:{data_files}")
else:
    data_file = data_files[0]
    file_like_object = BytesIO(data_file.read_bytes())
    df_data = pd.read_csv(file_like_object, on_bad_lines="skip")
    print(f"This is the Data File that was loaded: {data_file.name}")

    phrases_to_find = [
        "Raw Data for Passive Reference ROX",
        "Raw Data for Probe FAM-MGB",
        "Bkgd Data for Passive Reference ROX",
        "Bkgd Data for Probe FAM-MGB"
    ]
    file_like_object.seek(0)
     
    # Extract dataframes from each CSV file
    read_dataframes, date = reader.extract_dataframes_from_csv(file_like_object, phrases_to_find)

# at this point, we have loaded the assignment sheet and have sorted through the loaded data file to create a dict of dataframes 

# The premise of the code is that different viral panels require different thresholding
# So the user will specifiy command line arguments as per the ReadMe instructions
# and this portion of the code is meant to access the CI arguments and modify the threshold specified in the code
CLI_arg = sys.argv
# confirm that a threshold has been provided as a command line argument
if len(sys.argv) < 2:
    print("Please include the command line arguments when running analyze_run.py")
else:
    # Proceed with your script logic
    print("Threshold provided:", CLI_arg[1])


## Set up structure of the output folder - simplify into RESUTLS, QUALITY CONTROL, R&D
# make an output folder in your path's wd if it hasn't been made already
output_folder = f'output_{barcode_assignment}_{CLI_arg[1]}_v{software_version}'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Make R&D subfolder in the output folder in your path's wd if it hasn't been made already
rd_subfolder = os.path.join(output_folder, f'R&D_{barcode_assignment}')
if not os.path.exists(rd_subfolder):
    os.makedirs(rd_subfolder)

# Make RESULTS subfolder in the output folder in your path's wd if it hasn't been made already
res_subfolder = os.path.join(output_folder, f'RESULTS_{barcode_assignment}')
if not os.path.exists(res_subfolder):
    os.makedirs(res_subfolder)

# Make QUALITY CONTROLsubfolder in the output folder in your path's wd if it hasn't been made already
qc_subfolder = os.path.join(output_folder, f'QUALITY_CONTROL_{barcode_assignment}')
if not os.path.exists(qc_subfolder):
    os.makedirs(qc_subfolder)

# Make nested folders in QUALITY CONTROL for 'neg&pos controls' and 'viral assays'
npc_subfolder = os.path.join(qc_subfolder, f'QUALITY_CONTROL_OF_NEG_AND_POS_CONTROLS')
if not os.path.exists(npc_subfolder):
    os.makedirs(npc_subfolder)

va_subfolder = os.path.join(qc_subfolder, f'QUALITY_CONTROL_OF_VIRAL_ASSAYS')
if not os.path.exists(va_subfolder):
    os.makedirs(va_subfolder)


######################################################################################################################################################
# instantiate DataProcessor from norm.py
processor = DataProcessor()
# normalize the signal 
normalized_dataframes = processor.background_processing(read_dataframes)

# the output of DataProcessor is an array of dataframes where 1st is signal_norm and 2nd is ref_norm
# save these outputs to your output folder
normalized_dataframes['signal_norm'].to_csv(os.path.join(rd_subfolder, 'signal_norm.csv'), index=True)
normalized_dataframes['ref_norm'].to_csv(os.path.join(rd_subfolder, 'ref_norm.csv'), index=True)

######################################################################################################################################################
# instantiate DataMatcher from matcher.py
matcher = DataMatcher() 
assigned_norms, assigned_lists = matcher.assign_assays(assignment_files[0], normalized_dataframes['ref_norm'], normalized_dataframes['signal_norm'])

# save the output of assigned_norms array to your output folder
# 1st item is the assigned signal_norm_raw csv and 2nd is the assigned ref_norm_csv
assigned_norms['signal_norm_raw'].to_csv(os.path.join(rd_subfolder, 'assigned_signal_norm.csv'), index=True)
assigned_norms['ref_norm_raw'].to_csv(os.path.join(rd_subfolder, 'assigned_ref_norm.csv'), index=True)

# collect the assays/samples from the layout assays/samples in the assignment sheet (this extraction is done in matcher.py)
crRNA_assays = assigned_lists['assay_list']
samples_list = assigned_lists['samples_list']

######################################################################################################################################################
# instantiate ntcContaminationChecker from ntc_con_check.py
ntcCheck = ntcContaminationChecker()

# make a copy of assigned_signal_norm
assigned_signal_norm = pd.DataFrame(assigned_norms['signal_norm_raw']).copy() # make a copy of assigned_signal_norm dataframe
# create df of filtered assigned_signal_norm by applying the NTC check to remove any NTCs whose raw signal suggests contamination
assigned_signal_norm_with_NTC_check = ntcCheck.ntc_cont(assigned_signal_norm) # feed this into MedianSort

# temporarily save assigned_signal_norm_with_NTC_check
assigned_signal_norm_with_NTC_check.to_csv(os.path.join(rd_subfolder, 'assigned_signal_norm_with_NTC_check.csv'), index=True)


######################################################################################################################################################
# instantiate MedianSort from median_frame.py
median = MedianSort(crRNA_assays)
final_med_frames = median.create_median(assigned_signal_norm_with_NTC_check)

# temporarily print final_med_frames
#print(final_med_frames)

# Output needs to be rounded to 4 digits
rounded_final_med_frames = {}
# Define the number of decimals for rounding
decimals = 5

# Iterate through each row and column, round each value
for key, df in final_med_frames.items():
    rounded_df = pd.DataFrame(index=df.index, columns=df.columns)
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            # Round each value to the specified number of decimals
            rounded_df.iloc[i, j] = round(df.iloc[i, j], decimals)
    rounded_final_med_frames[key] = rounded_df

# Make subfolder in the output folder in your path's wd if it hasn't been made already
timepoints_subfolder = os.path.join(rd_subfolder, f'Quantitative Signal by Timepoint_{barcode_assignment}')
if not os.path.exists(timepoints_subfolder):
    os.makedirs(timepoints_subfolder)

# Save the dataframes per timepoint in subfolder timp
timepoints = list(rounded_final_med_frames.keys())
for i, t in enumerate(timepoints, start=1):
    filename = os.path.join(timepoints_subfolder, f't{i}_{barcode_assignment}.csv')
    csv = rounded_final_med_frames[t].to_csv(filename, index=True)


# since we want to explicitly manipulate t13_csv, it is helpful to have the t13 df referenced outside of the for loop
last_key = list(rounded_final_med_frames.keys())[-1]
t13_dataframe_orig = rounded_final_med_frames[last_key]
t13_dataframe_copy1 = pd.DataFrame(t13_dataframe_orig).copy()
t13_dataframe_copy2 = pd.DataFrame(t13_dataframe_orig).copy()

# at this point, we have created a t1 thru t13 dataframe and exported all these dataframes as csv files in our output folder
# now we need to threshold the t13 csv and mark signals >= threshold as positive and < threshold as negative

######################################################################################################################################################
# instantiate Thresholder from threshold.py
thresholdr = Thresholder()
unique_crRNA_assays = list(set(crRNA_assays))

# apply the NTC thresholding to the t13_dataframe to produce a new dataframe with the positive/negative denotation
# and save the file to your working directory
ntc_thresholds_output, t13_hit_output = thresholdr.raw_thresholder(unique_crRNA_assays, assigned_signal_norm_with_NTC_check, t13_dataframe_copy1, CLI_arg[1])

# round the NTC thresholds to 5 places
decimals = 5 # Define the number of decimals for rounding
rounded_ntc_thresholds_output = pd.DataFrame(index=ntc_thresholds_output.index, columns=ntc_thresholds_output.columns) # Iterate through each row and column, round each value
for i in range(len(ntc_thresholds_output.index)):
    for j in range(len(ntc_thresholds_output.columns)):
        # Round each value to the specified number of decimals
        rounded_ntc_thresholds_output.iloc[i, j] = round(ntc_thresholds_output.iloc[i, j], decimals)
# save the NTC thresholds to a csv after flags are added below in Flagger()


# make copies of t13_hit_output csv for downstream summaries and quality control checks
t13_hit_output_copy1 = pd.DataFrame(t13_hit_output).copy() # make a copy of t13_hit_output # used in ndc qual check
t13_hit_output_copy2 = pd.DataFrame(t13_hit_output).copy() # make a copy of t13_hit_output # used in cpc qual check
t13_hit_output_copy3 = pd.DataFrame(t13_hit_output).copy() # make a copy of t13_hit_output # used in rnasep qual check
t13_hit_output_copy4 = pd.DataFrame(t13_hit_output).copy() # make a copy of t13_hit_output # used in t13_hit_binary_output generation
t13_hit_output_copy5 = pd.DataFrame(t13_hit_output).copy() # make a copy of t13_hit_output
t13_hit_output_copy6 = pd.DataFrame(t13_hit_output).copy() # make a copy of t13_hit_output
# save t13_hit_output as Results_Summary after flags are added below in Flagger()

######################################################################################################################################################
# instantiate NTC_Normalized from ntcnorm.py
ntcNorm = Normalized()
# apply ntc_normalizr to the t13_dataframe to produce a new dataframe with all values divided by the mean NTC for that assay
t13_quant_norm = ntcNorm.normalizr(t13_dataframe_copy2)
# round the NTC normalized quantitative results file
decimals = 5 # Define the number of decimals for rounding
rounded_t13_quant_norm = pd.DataFrame(index=t13_quant_norm.index, columns=t13_quant_norm.columns) # Iterate through each row and column, round each value
for i in range(len(t13_quant_norm.index)):
    for j in range(len(t13_quant_norm.columns)):
        # Round each value to the specified number of decimals
        rounded_t13_quant_norm.iloc[i, j] = round(t13_quant_norm.iloc[i, j], decimals)

# append the totals row from summary_samples_df to the bottom of rounded_t13_quant_norm
sum_row = t13_hit_output.loc[['Summary']]
rounded_t13_quant_norm = pd.concat((rounded_t13_quant_norm, sum_row), ignore_index=False)

# save the rounded_t13_quant_norm as NTC_Quant_Results_Summary after flags are added below in Flagger()

######################################################################################################################################################
# instantiate Binary_Converter from binary_results.py
binary_num_converter = Binary_Converter()
# apply hit_numeric_conv to the the t13_hit_output to produce a new dataframe with all pos/neg converted to binary 1/0 output
t13_hit_binary_output = binary_num_converter.hit_numeric_conv(t13_hit_output_copy4)
# save t13_hit_binary_output after flags are added below in Flagger()

# make copies of t13_hit_binary_output for downstream utilization in coinf check and assay level eval
t13_hit_binary_output_copy1 = pd.DataFrame(t13_hit_binary_output).copy() # used in coninf check
t13_hit_binary_output_copy2 = pd.DataFrame(t13_hit_binary_output).copy() 
 

######################################################################################################################################################
# instantiate Summarized from summary.py
summary = Summarized()
# apply summarizer to the t13_dataframe to produce a new dataframe tabulating all of the positive samples
summary_samples_df = summary.summarizer(t13_hit_output)
# save summary_samples_df as Positives Summary after flags are added in Flagger() below

######################################################################################################################################################
# instantiate Assay_QC_Score
assayScorer = Assay_QC_Score()
# take in t13_hit_binary_output as the df to build off of 
QC_score_per_assay_df = assayScorer.assay_level_score(t13_hit_binary_output)

# save the QC_score_per_assay_df to the va_subfolder (viral assays subfolder which is nested inside the QC subfolder)
assay_lvl_QC_score_file_path = os.path.join(va_subfolder, f'Assay_Performance_QC_Test_Results_{barcode_assignment}.csv')
QC_score_per_assay_df.to_csv(assay_lvl_QC_score_file_path, index=True)

# copy the Assay-Level Test Explanation pdf to the va_subfolder
assay_test_expl_source_file = 'Assay-Level QC Test Explanation.pdf'
assay_test_expl_output_file = os.path.join(va_subfolder, 'Assay-Level QC Test Explanation.pdf')
shutil.copy(assay_test_expl_source_file, assay_test_expl_output_file)

print(f"The four quality control tests to evaluate assay performance are complete. Their results have been saved to the folder, {va_subfolder}")

######################################################################################################################################################
# instantiate Qual_Ctrl_Checks from qual-checks.py
qual_checks = Qual_Ctrl_Checks()

# Define the path for the PDF file in the qc_subfolder
npc_pdf_file_path = os.path.join(npc_subfolder, f'Quality_Control_Report_{barcode_assignment}.pdf')
doc = SimpleDocTemplate(npc_pdf_file_path, pagesize=A4,
                        leftMargin=50, rightMargin=50, topMargin=50, bottomMargin=50)

# Define text styles
styles = getSampleStyleSheet()
style = styles['Normal']
header_style = styles['Heading2']
header_style.fontName = "Times-Roman"
header_style.fontSize = 12
header_style.textColor = (0, 0, 0.5)  # Set header color to blue
style.fontName = 'Times-Roman'
style.fontSize = 12

content = []


### BUILD QC Excel file
# Set up QC sheet names
QC_sheet_names = [
    "Positive NDC Samples",
    "Negative CPC Samples",
    "Negative RNaseP Samples",
    "Positive NTC Samples",
    "Potentially Co-infected Samples", 
    "Invalid Samples"
]

qc_output_file_path = os.path.join(npc_subfolder, f"QC_{barcode_assignment}.xlsx") #

try:
    with pd.ExcelWriter(qc_output_file_path, engine="xlsxwriter") as writer:

        ### (1) NDC CHECK

        # apply ndc_check to the t13_hit_output df to generate a list of all ndc positive assays
        ndc_positives_df = qual_checks.ndc_check(t13_hit_output_copy1)

        # When converting the ndc_positives df into a CSV, check if ndc_positives_df is empty and, if so, modify the CSV produced 
        if ndc_positives_df.empty:
            # If empty, create a DataFrame with the custom message and save it to CSV
            empty_message_df = pd.DataFrame({"Message": ["For all viral assays tested in this experiment, all NDC samples test negative."]})
            empty_message_df.to_excel(writer, sheet_name=QC_sheet_names[0], index=False)
            print(f"File generated with message in {QC_sheet_names[0]} at {qc_output_file_path}")
        else:
            # If ndc_positives_df is not empty, save the DataFrame as is
            ndc_positives_df.to_excel(writer, sheet_name=QC_sheet_names[0], index=True)
            print(f"File generated with data in {QC_sheet_names[0]} at {qc_output_file_path}")

    
        # Add NDC Check Header and Paragraphs
        content.append(Paragraph("1. Evaluation of No Detect Control (NDC) Contamination", header_style))
        content.append(Spacer(1, 0.2 * inch))

        if not ndc_positives_df.empty:
            text_content = [
                f"Please consult NDC_Check_{barcode_assignment}.csv to see the initial evaluation of the NDC negative controls tested in this experiment. In this file, assays are flagged for which the NDC samples have tested positive, after being thresholded against the assay-specific NTC mean.",
                "If any of the NDC samples show a positive result for any assay, then that assay should be evaluated for contamination with nucleases likely at the sample mastermix preparation step in the experimental workflow. However, other sources for NDC contamination may exist.\n",
                "Please be advised to check the output files as well."
            ]
        else:
            text_content = [
                "Since none of the NDCs ran in this experiment appear positive, there is likely no NDC contamination.",
                "Please check the output files as well."
            ]

        for line in text_content:
            content.append(Paragraph(line, style))
            content.append(Spacer(1, 0.1 * inch))

        ### (2) CPC Check

        ## apply cpc_check to the t13_hit_output df to generate a list of all cpc negative assays
        cpc_negatives_df = qual_checks.cpc_check(t13_hit_output_copy2)

        # When converting the cpc_negatives_df into a CSV, check if cpc_negatives_df is empty and, if so, modify the CSV produced 
        if cpc_negatives_df.empty:
            # If empty, create a DataFrame with the custom message and save it to CSV
            empty_message_df = pd.DataFrame({"Message": ["For all viral assays tested in this experiment, all CPC samples test positive."]})
            empty_message_df.to_excel(writer, sheet_name=QC_sheet_names[1], index=False)
            print(f"File generated with message in {QC_sheet_names[1]} at {qc_output_file_path}")
        else:
            # If cpc_negatives_df is not empty, save the DataFrame as is
            cpc_negatives_df.to_excel(writer, sheet_name=QC_sheet_names[1], index=True)
            print(f"File generated with data in {QC_sheet_names[1]} at {qc_output_file_path}")


        # Add CPC Check Header and Paragraphs
        content.append(Paragraph("2. Evaluation of Combined Positive Control (CPC) Validity", header_style))
        content.append(Spacer(1, 0.2 * inch))

        if not cpc_negatives_df.empty:
            text_content = [
                f"Please consult CPC_Check_{barcode_assignment}.csv to see the initial evaluation of the CPC positive controls tested in this experiment. In this file, assays are flagged for which the CPC samples have tested negative, after being thresholded against the assay-specific NTC mean.",
                "If any of the CPC samples show a negative result for any assay excluding the 'no-crRNA' negative control assay, then that assay should be considered invalid for this experiment.",
                "Please be advised to check the output files as well."
            ]
        else:
            text_content = [
                "Warning: First verify that your experiment included a CPC sample. If yes, proceed to the following CPC analysis.",
                "After thresholding against the NTC, the CPC(s) appears as positive for all crRNA assays tested. However, it is expected for the CPC(s) to test as negative for 'no-crRNA' assay. There may be possible contamination of the 'no-crRNA' assay.",
                "Please be advised to check the output files as well."
            ]

        for line in text_content:
            content.append(Paragraph(line, style))
            content.append(Spacer(1, 0.1 * inch))


        ### (3) RNaseP Check

        ## apply rnasep_check to the t13_hit_output df to generate a list of all rnasep negative samples
        rnasep_df = qual_checks.rnasep_check(t13_hit_output_copy3)
        rnasep_df_heatmap = rnasep_df.copy()

        # When converting the cpc_negatives_df into a CSV, check if cpc_negatives_df is empty and, if so, modify the CSV produced 
        if rnasep_df.empty:
            # If empty, create a DataFrame with the custom message and save it to CSV
            empty_message_df = pd.DataFrame({"Message": ["Please verify that you have included an RNaseP viral assay in your CARMEN experiment. If you have, all samples test postive for the RNaseP assay."]})
            empty_message_df.to_excel(writer, sheet_name=QC_sheet_names[2], index=False)
            print(f"File generated with message in {QC_sheet_names[2]} at {qc_output_file_path}")
        else:
            # If cpc_negatives_df is not empty, save the DataFrame as is
            rnasep_df.to_excel(writer, sheet_name=QC_sheet_names[2], index=True)
            print(f"File generated with data in {QC_sheet_names[2]} at {qc_output_file_path}")

        # Add RNaseP Check Header and Paragraphs
        content.append(Paragraph("3. Evaluation of Human Samples for the Internal Control (RNaseP)", header_style))
        content.append(Spacer(1, 0.2 * inch))
        
        if not rnasep_df.empty:
            text_content = [
                "Warning: First verify that your experiment included a RNaseP assay. If yes, proceed to the following RNaseP analysis.",
                f"Please consult RNaseP_Check_{barcode_assignment}.csv to see which samples are negative for the RNaseP assay(s). In this file, the samples that appear negative for the RNaseP assays have been flagged after thresholding against the NTC. The negative controls (NTC and NDC) are expected to be negative for the RNaseP assay and should be listed here (if you have included them in this experiment). All other samples should be evaluated for being negative for the RNaseP assay.",
            
                "Possible reasons for a sample testing negative for the RNaseP assay:",
                "(A) If the sample is negative for all assays (including RNaseP), then the most plausible hypothesis is that the viral extraction protocol used in this experiment needs to be examined. For optimal results, the extraction must be compatible with the Standard Operating Procedure (SOP) advised by the CARMEN team in the Sabeti Lab.",
                "** Note: If the sample is negative for RNaseP and ALL other crRNA assays tested in this experiment, the sample should be rendered invalid.",
                
                "(B) If the sample is negative for RNaseP BUT positive for any other viral crRNA assay (excluding RNaseP or no-crRNA), then the most plausible hypothesis is that the sample’s viral titer may be too high compared to its RNaseP titer. This, thereby, renders the system possibly unable to detect RNaseP, leading to the sample testing negative for RNaseP.",
                "** Note: If the sample is negative for RNaseP but positive for any other viral crRNA assay (excluding RNaseP or no-crRNA) tested in this experiment, the sample can still be included in the final results.",

                "(C) The source sample may have insufficient material, leading to a negative RNaseP signal and an invalid sample result.",

                "Please be advised to check the output files as well."

            ]
        else:
            text_content = [
                "Warning: First verify that your experiment included a RNaseP assay. If yes, proceed to the following RNaseP analysis.",
                "All samples (including negative controls) have tested positive for the RNaseP assay(s) tested in this experiment. However, the assay(s) for RNaseP internal control should test negative for the NTC and NDC negative control.",
                "There are a few different reasons that all samples test positive for RNaseP. The most plausible hypothesis is that there is RNaseP contamination in this experiment. Precaution is advised to mitigate contamination avenues, especially at the RT-PCR (nucleic acid amplification) stage.",
                "Please be advised to check the output files as well."
            ]

        for line in text_content:
            content.append(Paragraph(line, style))
            content.append(Spacer(1, 0.1 * inch))

        ### (4) NTC Check

        ## apply ntc_check to the t13_hit_output df to generate a list of all ntc positive assays
        assigned_signal_norm_2 = pd.DataFrame(assigned_norms['signal_norm_raw']).copy() # make a copy of assigned_signal_norm dataframe
        high_raw_ntc_signal_df = qual_checks.ntc_check(assigned_signal_norm_2)

        # When converting the high_raw_ntc_signal_df into a CSV, check if high_raw_ntc_signal_df is empty and, if so, modify the CSV produced 
        if high_raw_ntc_signal_df.empty:
            # If empty, create a DataFrame with the custom message and save it to CSV
            empty_message_df = pd.DataFrame({"Message": ["For all viral assays tested in this experiment, there are no NTC samples which appear as contaminated."]})
            empty_message_df.to_excel(writer, sheet_name=QC_sheet_names[3], index=False)
            print(f"File generated with message in {QC_sheet_names[3]} at {qc_output_file_path}")
        else:
            # If cpc_negatives_df is not empty, save the DataFrame as is
            high_raw_ntc_signal_df.to_excel(writer, sheet_name=QC_sheet_names[3], index=True)
            print(f"File generated with data in {QC_sheet_names[3]} at {qc_output_file_path}")


        # Add NTC Check Header and Paragraphs
        content.append(Paragraph("4. Evaluation of No Target Control (NTC) Contamination", header_style))
        content.append(Spacer(1, 0.2 * inch))

        if not high_raw_ntc_signal_df.empty:
            text_content = [
                f"Please consult NTC_Contamination_Check_{barcode_assignment}.csv to see which NTC samples may be potentially contaminated.",
                "This file contains a list of samples that have a raw fluorescence signal above 0.5 a.u. These samples are being flagged for having a higher than normal signal for an NTC sample. The range for typical raw fluorescence signal for an NTC sample is between 0.1 and 0.5 a.u.",
                "Please be advised to check the output files to further evaluate potential NTC contamination."
            ]
        else:
            text_content = [
                "The raw fluorescence signal for each NTC sample across all crRNA assays tested in this experiment appears to be within the normal range of 0.1 and 0.5 a.u. Risk of NTC contamination is low.",
                "Please be advised to check the output files as well."
            ]
            
        for line in text_content:
            content.append(Paragraph(line, style))
            content.append(Spacer(1, 0.1 * inch))


        ### (5) Co-Infection Check

        ## apply coinfection check to t13_hit_binary_output to generate list of all samples that are positive for multiple assays
        coinfection_df = qual_checks.coinf_check(t13_hit_binary_output_copy1)

        # define file path for csv
        coinfection_df_file_path = os.path.join(npc_subfolder, f'Coinfection_Check_{barcode_assignment}.csv')

        # When converting the coinfection_df into a CSV, check if coinfection_df is empty and, if so, modify the CSV produced 
        if coinfection_df.empty:
            # If empty, create a DataFrame with the custom message and save it to CSV
            empty_message_df = pd.DataFrame({"Message": ["For all viral assays tested in this experiment, there are no samples which appear as potentially co-infected."]})
            empty_message_df.to_excel(writer, sheet_name=QC_sheet_names[4], index=False)
            print(f"File generated with message in {QC_sheet_names[4]} at {qc_output_file_path}")
        else:
            # If coinfection_df is not empty, save the DataFrame as is
            coinfection_df.to_excel(writer, sheet_name=QC_sheet_names[4], index=True)
            print(f"File generated with data in {QC_sheet_names[4]} at {qc_output_file_path}")

        # Add NTC Check Header and Paragraphs
        content.append(Paragraph("5. Evaluation of Potential Co-Infected Samples", header_style))
        content.append(Spacer(1, 0.2 * inch))

        if not coinfection_df.empty:
            text_content = [
            f"Please consult Codetection_Check_{barcode_assignment}.csv to see which samples may be potentially co-infected.",
                "A preliminary evaluation for co-infection of a given sample against all tested assays has been completed:",
                "   (A) If you have included Combined Positive Controls (CPCs) in this experiment, as recommended, these positive controls should be identified and listed among the flagged samples. CPCs are expected to show a “co-detection” with ALL of the assays being tested in this experiment.",
                "   (B) Samples are not flagged as “co-detected” based on positivity with RNaseP and a second assay. For a sample to be flagged during this Co-detection Check, it must test positive for at least two assays, excluding RNaseP.",
                "   (C) All other flagged samples should be further evaluated for potential co-infection.",
                "Please be advised to check the output files to further evaluate potential co-infection."
            ]
        else:
            text_content = [
                f"Please consult Codetection_Check_{barcode_assignment}.csv to see which samples may be potentially co-infected.",
                "A preliminary evaluation for co-infection of a given sample against all tested assays has been completed:",
                "   (A) If you have included Combined Positive Controls (CPCs) in this experiment, as recommended, these positive controls should be identified and listed among the flagged samples. CPCs are expected to show a “co-detection” with ALL of the assays being tested in this experiment.",
                "   (B) Samples are not flagged as “co-detected” based on positivity with RNaseP and a second assay. For a sample to be flagged during this Co-detection Check, it must test positive for at least two assays, excluding RNaseP.",
                "   (C) All other flagged samples should be further evaluated for potential co-infection.",
                "Please be advised to check the output files to further evaluate potential co-infection."    
            ]
            
        for line in text_content:
            content.append(Paragraph(line, style))
            content.append(Spacer(1, 0.1 * inch))

        # Build the PDF with the collected Flowables
        doc.build(content)

        ## (6) No-crRNA Assay Check 

        ## apply no_crrna check to t13_hit_binary_output to generate list of all samples that are positive for the no-crRNA assay(s)
        fail_nocrRNA_check_df = qual_checks.no_crrna_check(t13_hit_binary_output_copy1)

        # define file path for csv
        fail_nocrRNA_check_df_file_path = os.path.join(npc_subfolder, f'No_crRNA_Assay_Check_{barcode_assignment}.csv')

        # When converting the coinfection_df into a CSV, check if coinfection_df is empty and, if so, modify the CSV produced 
        if fail_nocrRNA_check_df.empty:
            # If empty, create a DataFrame with the custom message and save it to CSV
            empty_message_df = pd.DataFrame({"Message": ["All samples have tested against the no-crRNA assay. Thus, there are no samples which must be invalidated."]})
            empty_message_df.to_excel(writer, sheet_name=QC_sheet_names[5], index=False)
            print(f"File generated with message in {QC_sheet_names[5]} at {qc_output_file_path}")
        else:
            # If coinfection_df is not empty, save the DataFrame as is
            fail_nocrRNA_check_df.to_excel(writer, sheet_name=QC_sheet_names[5], index=True)
            print(f"File generated with data in {QC_sheet_names[5]} at {qc_output_file_path}")

except Exception as e:
    print(f'Error with QC Excel file generation: {e}. Results are saved as 5 individual csv files.')

    ## (1) NDC CHECK

    # apply ndc_check to the t13_hit_output df to generate a list of all ndc positive assays
    ndc_positives_df = qual_checks.ndc_check(t13_hit_output_copy1)

    # define file path for csv
    ndc_positives_df_file_path = os.path.join(npc_subfolder, f'NDC_Check_{barcode_assignment}.csv')

    # When converting the ndc_positives df into a CSV, check if ndc_positives_df is empty and, if so, modify the CSV produced 
    if ndc_positives_df.empty:
        # If empty, create a DataFrame with the custom message and save it to CSV
        empty_message_df = pd.DataFrame({"Message": ["For all viral assays tested in this experiment, all NDC samples test negative."]})
        empty_message_df.to_csv(ndc_positives_df_file_path, index=False, header=False)
        print(f"CSV created with message at {ndc_positives_df_file_path}")
    else:
        # If ndc_positives_df is not empty, save the DataFrame as is
        ndc_positives_df.to_csv(ndc_positives_df_file_path, index=True)
        print(f"CSV created with data at {ndc_positives_df_file_path}")

    # Add NDC Check Header and Paragraphs
    content.append(Paragraph("1. Evaluation of No Detect Control (NDC) Contamination", header_style))
    content.append(Spacer(1, 0.2 * inch))

    if not ndc_positives_df.empty:
        text_content = [
            f"Please consult NDC_Check_{barcode_assignment}.csv to see the initial evaluation of the NDC negative controls tested in this experiment. In this file, assays are flagged for which the NDC samples have tested positive, after being thresholded against the assay-specific NTC mean.",
            "If any of the NDC samples show a positive result for any assay, then that assay should be evaluated for contamination with nucleases likely at the sample mastermix preparation step in the experimental workflow. However, other sources for NDC contamination may exist.\n",
            "Please be advised to check the output files as well."
        ]
    else:
        text_content = [
            "Since none of the NDCs ran in this experiment appear positive, there is likely no NDC contamination.",
            "Please check the output files as well."
        ]

    for line in text_content:
        content.append(Paragraph(line, style))
        content.append(Spacer(1, 0.1 * inch))

    ## (2) CPC Check

    ## apply cpc_check to the t13_hit_output df to generate a list of all cpc negative assays
    cpc_negatives_df = qual_checks.cpc_check(t13_hit_output_copy2)

    # define file path for csv
    cpc_negatives_df_file_path = os.path.join(npc_subfolder, f'CPC_Check_{barcode_assignment}.csv')

    # When converting the cpc_negatives_df into a CSV, check if cpc_negatives_df is empty and, if so, modify the CSV produced 
    if cpc_negatives_df.empty:
        # If empty, create a DataFrame with the custom message and save it to CSV
        empty_message_df = pd.DataFrame({"Message": ["For all viral assays tested in this experiment, all CPC samples test positive."]})
        empty_message_df.to_csv(cpc_negatives_df_file_path, index=False, header=False)
        print(f"CSV created with message at {cpc_negatives_df_file_path}")
    else:
        # If cpc_negatives_df is not empty, save the DataFrame as is
        cpc_negatives_df.to_csv(cpc_negatives_df_file_path, index=True)
        print(f"CSV created with data at {cpc_negatives_df_file_path}")


    # Add CPC Check Header and Paragraphs
    content.append(Paragraph("2. Evaluation of Combined Positive Control (CPC) Validity", header_style))
    content.append(Spacer(1, 0.2 * inch))

    if not cpc_negatives_df.empty:
        text_content = [
            f"Please consult CPC_Check_{barcode_assignment}.csv to see the initial evaluation of the CPC positive controls tested in this experiment. In this file, assays are flagged for which the CPC samples have tested negative, after being thresholded against the assay-specific NTC mean.",
            "If any of the CPC samples show a negative result for any assay excluding the 'no-crRNA' negative control assay, then that assay should be considered invalid for this experiment.",
            "Please be advised to check the output files as well."
        ]
    else:
        text_content = [
            "Warning: First verify that your experiment included a CPC sample. If yes, proceed to the following CPC analysis.",
            "After thresholding against the NTC, the CPC(s) appears as positive for all crRNA assays tested. However, it is expected for the CPC(s) to test as negative for 'no-crRNA' assay. There may be possible contamination of the 'no-crRNA' assay.",
            "Please be advised to check the output files as well."
        ]

    for line in text_content:
        content.append(Paragraph(line, style))
        content.append(Spacer(1, 0.1 * inch))


    ## (3) RNaseP Check

    ## apply rnasep_check to the t13_hit_output df to generate a list of all rnasep negative samples
    rnasep_df = qual_checks.rnasep_check(t13_hit_output_copy3)
    rnasep_df_heatmap = rnasep_df.copy()

    # define file path for csv
    rnasep_df_file_path = os.path.join(npc_subfolder, f'RNaseP_Check_{barcode_assignment}.csv')

    # When converting the cpc_negatives_df into a CSV, check if cpc_negatives_df is empty and, if so, modify the CSV produced 
    if rnasep_df.empty:
        # If empty, create a DataFrame with the custom message and save it to CSV
        empty_message_df = pd.DataFrame({"Message": ["Please verify that you have included an RNaseP viral assay in your CARMEN experiment. If you have, all samples test postive for the RNaseP assay."]})
        empty_message_df.to_csv(rnasep_df_file_path, index=False, header=False)
        print(f"CSV created with message at {rnasep_df_file_path}")
    else:
        # If cpc_negatives_df is not empty, save the DataFrame as is
        rnasep_df.to_csv(rnasep_df_file_path, index=True)
        print(f"CSV created with data at {rnasep_df_file_path}")

    # Add RNaseP Check Header and Paragraphs
    content.append(Paragraph("3. Evaluation of Human Samples for the Internal Control (RNaseP)", header_style))
    content.append(Spacer(1, 0.2 * inch))

    if not rnasep_df.empty:
        text_content = [
            "Warning: First verify that your experiment included a RNaseP assay. If yes, proceed to the following RNaseP analysis.",
            f"Please consult RNaseP_Check_{barcode_assignment}.csv to see which samples are negative for the RNaseP assay(s). In this file, the samples that appear negative for the RNaseP assays have been flagged after thresholding against the NTC. The negative controls (NTC and NDC) are expected to be negative for the RNaseP assay and should be listed here (if you have included them in this experiment). All other samples should be evaluated for being negative for the RNaseP assay.",
        
            "Possible reasons for a sample testing negative for the RNaseP assay:",
            "(A) If the sample is negative for all assays (including RNaseP), then the most plausible hypothesis is that the viral extraction protocol used in this experiment needs to be examined. For optimal results, the extraction must be compatible with the Standard Operating Procedure (SOP) advised by the CARMEN team in the Sabeti Lab.",
            "** Note: If the sample is negative for RNaseP and ALL other crRNA assays tested in this experiment, the sample should be rendered invalid.",
            
            "(B) If the sample is negative for RNaseP BUT positive for any other viral crRNA assay (excluding RNaseP or no-crRNA), then the most plausible hypothesis is that the sample’s viral titer may be too high compared to its RNaseP titer. This, thereby, renders the system possibly unable to detect RNaseP, leading to the sample testing negative for RNaseP.",
            "** Note: If the sample is negative for RNaseP but positive for any other viral crRNA assay (excluding RNaseP or no-crRNA) tested in this experiment, the sample can still be included in the final results.",

            "(C) The source sample may have insufficient material, leading to a negative RNaseP signal and an invalid sample result.",

            "Please be advised to check the output files as well."

        ]
    else:
        text_content = [
            "Warning: First verify that your experiment included a RNaseP assay. If yes, proceed to the following RNaseP analysis.",
            "All samples (including negative controls) have tested positive for the RNaseP assay(s) tested in this experiment. However, the assay(s) for RNaseP internal control should test negative for the NTC and NDC negative control.",
            "There are a few different reasons that all samples test positive for RNaseP. The most plausible hypothesis is that there is RNaseP contamination in this experiment. Precaution is advised to mitigate contamination avenues, especially at the RT-PCR (nucleic acid amplification) stage.",
            "Please be advised to check the output files as well."
        ]

    for line in text_content:
        content.append(Paragraph(line, style))
        content.append(Spacer(1, 0.1 * inch))

    ## (4) NTC Check

    ## apply ntc_check to the t13_hit_output df to generate a list of all ntc positive assays
    assigned_signal_norm_2 = pd.DataFrame(assigned_norms['signal_norm_raw']).copy() # make a copy of assigned_signal_norm dataframe
    high_raw_ntc_signal_df = qual_checks.ntc_check(assigned_signal_norm_2)

    # define file path for csv
    high_raw_ntc_signal_df_file_path = os.path.join(npc_subfolder, f'NTC_Contamination_Check_{barcode_assignment}.csv')

    # When converting the high_raw_ntc_signal_df into a CSV, check if high_raw_ntc_signal_df is empty and, if so, modify the CSV produced 
    if high_raw_ntc_signal_df.empty:
        # If empty, create a DataFrame with the custom message and save it to CSV
        empty_message_df = pd.DataFrame({"Message": ["For all viral assays tested in this experiment, there are no NTC samples which appear as contaminated."]})
        empty_message_df.to_csv(high_raw_ntc_signal_df_file_path, index=False, header=False)
        print(f"CSV created with message at {high_raw_ntc_signal_df_file_path}")
    else:
        # If cpc_negatives_df is not empty, save the DataFrame as is
        high_raw_ntc_signal_df.to_csv(high_raw_ntc_signal_df_file_path, index=True)
        print(f"CSV created with data at {high_raw_ntc_signal_df_file_path}")


    # Add NTC Check Header and Paragraphs
    content.append(Paragraph("4. Evaluation of No Target Control (NTC) Contamination", header_style))
    content.append(Spacer(1, 0.2 * inch))

    if not high_raw_ntc_signal_df.empty:
        text_content = [
            f"Please consult NTC_Contamination_Check_{barcode_assignment}.csv to see which NTC samples may be potentially contaminated.",
            "This file contains a list of samples that have a raw fluorescence signal above 0.5 a.u. These samples are being flagged for having a higher than normal signal for an NTC sample. The range for typical raw fluorescence signal for an NTC sample is between 0.1 and 0.5 a.u.",
            "Please be advised to check the output files to further evaluate potential NTC contamination."
        ]
    else:
        text_content = [
            "The raw fluorescence signal for each NTC sample across all crRNA assays tested in this experiment appears to be within the normal range of 0.1 and 0.5 a.u. Risk of NTC contamination is low.",
            "Please be advised to check the output files as well."
        ]
        
    for line in text_content:
        content.append(Paragraph(line, style))
        content.append(Spacer(1, 0.1 * inch))


    ## (5) Co-Infection Check

    ## apply coinfection check to t13_hit_binary_output to generate list of all samples that are positive for multiple assays
    coinfection_df = qual_checks.coinf_check(t13_hit_binary_output_copy1)

    # define file path for csv
    coinfection_df_file_path = os.path.join(npc_subfolder, f'Coinfection_Check_{barcode_assignment}.csv')

    # When converting the coinfection_df into a CSV, check if coinfection_df is empty and, if so, modify the CSV produced 
    if coinfection_df.empty:
        # If empty, create a DataFrame with the custom message and save it to CSV
        empty_message_df = pd.DataFrame({"Message": ["For all viral assays tested in this experiment, there are no samples which appear as potentially co-infected."]})
        empty_message_df.to_csv(coinfection_df_file_path, index=False, header=False)
        print(f"CSV created with message at {coinfection_df_file_path}")
    else:
        # If coinfection_df is not empty, save the DataFrame as is
        coinfection_df.to_csv(coinfection_df_file_path, index=True)
        print(f"CSV created with data at {coinfection_df_file_path}")

    # Add NTC Check Header and Paragraphs
    content.append(Paragraph("5. Evaluation of Potential Co-Infected Samples", header_style))
    content.append(Spacer(1, 0.2 * inch))

    if not coinfection_df.empty:
        text_content = [
        f"Please consult Codetection_Check_{barcode_assignment}.csv to see which samples may be potentially co-infected.",
            "A preliminary evaluation for co-infection of a given sample against all tested assays has been completed:",
            "   (A) If you have included Combined Positive Controls (CPCs) in this experiment, as recommended, these positive controls should be identified and listed among the flagged samples. CPCs are expected to show a “co-detection” with ALL of the assays being tested in this experiment.",
            "   (B) Samples are not flagged as “co-detected” based on positivity with RNaseP and a second assay. For a sample to be flagged during this Co-detection Check, it must test positive for at least two assays, excluding RNaseP.",
            "   (C) All other flagged samples should be further evaluated for potential co-infection.",
            "Please be advised to check the output files to further evaluate potential co-infection."
        ]
    else:
        text_content = [
            f"Please consult Codetection_Check_{barcode_assignment}.csv to see which samples may be potentially co-infected.",
            "A preliminary evaluation for co-infection of a given sample against all tested assays has been completed:",
            "   (A) If you have included Combined Positive Controls (CPCs) in this experiment, as recommended, these positive controls should be identified and listed among the flagged samples. CPCs are expected to show a “co-detection” with ALL of the assays being tested in this experiment.",
            "   (B) Samples are not flagged as “co-detected” based on positivity with RNaseP and a second assay. For a sample to be flagged during this Co-detection Check, it must test positive for at least two assays, excluding RNaseP.",
            "   (C) All other flagged samples should be further evaluated for potential co-infection.",
            "Please be advised to check the output files to further evaluate potential co-infection."    
        ]
        
    for line in text_content:
        content.append(Paragraph(line, style))
        content.append(Spacer(1, 0.1 * inch))

    # Build the PDF with the collected Flowables
    doc.build(content)

    ## (6) No-crRNA Assay Check 

    ## apply no_crrna check to t13_hit_binary_output to generate list of all samples that are positive for the no-crRNA assay(s)
    fail_nocrRNA_check_df = qual_checks.no_crrna_check(t13_hit_binary_output_copy1)

    # define file path for csv
    fail_nocrRNA_check_df_file_path = os.path.join(npc_subfolder, f'No_crRNA_Assay_Check_{barcode_assignment}.csv')

    # When converting the coinfection_df into a CSV, check if coinfection_df is empty and, if so, modify the CSV produced 
    if fail_nocrRNA_check_df.empty:
        # If empty, create a DataFrame with the custom message and save it to CSV
        empty_message_df = pd.DataFrame({"Message": ["All samples have tested against the no-crRNA assay. Thus, there are no samples which must be invalidated."]})
        empty_message_df.to_csv(fail_nocrRNA_check_df_file_path, index=False, header=False)
        print(f"CSV created with message at {fail_nocrRNA_check_df_file_path}")
    else:
        # If coinfection_df is not empty, save the DataFrame as is
        fail_nocrRNA_check_df.to_csv(fail_nocrRNA_check_df_file_path, index=True)
        print(f"CSV created with data at {fail_nocrRNA_check_df_file_path}")

 
###################################################################################################################################################### 
# instantiate Flagger from flags.py
flagger = Flagger()

invalid_assays, invalid_samples, flagged_files, processed_samples = flagger.assign_flags(fail_nocrRNA_check_df, high_raw_ntc_signal_df, rnasep_df_heatmap, QC_score_per_assay_df, t13_hit_output, rounded_t13_quant_norm, summary_samples_df, rounded_ntc_thresholds_output, t13_hit_binary_output)

fl_t13_hit_output = flagged_files[0] # Results_Summary
fl_rounded_t13_quant_norm = flagged_files[1] # NTC_Normalized_Quantitative_Results_Summary
fl_summary_samples_df = flagged_files[2] # Positives_Summary
fl_rounded_ntc_thresholds_output = flagged_files[3] # NTC_thresholds
fl_t13_hit_binary_output = flagged_files[4] # 't13__{barcode_assignment}_hit_binary

# any last formatting changes to dataframes
fl_t13_hit_binary_output.columns = fl_t13_hit_binary_output.columns.str.upper()

# SAVE all the files to their respective output folders
sheet_names = [
    "CARMEN_Hit_Results",
    "NTC_Normalized_Quant_Results",
    "Summary_of_Positive_Samples",
    "NTC_thresholds",
    "Binary_Results"
]

dataframes = [
    fl_t13_hit_output,
    fl_rounded_t13_quant_norm,
    fl_summary_samples_df,
    fl_rounded_ntc_thresholds_output,
    fl_t13_hit_binary_output
]

output_file_path = os.path.join(res_subfolder, f"RESULTS_{barcode_assignment}_{CLI_arg[1]}.xlsx") #

try: 
    # save all DataFrames to a single Excel file
    with pd.ExcelWriter(output_file_path, engine="xlsxwriter") as writer:
        # store the indices of cells with the "POSITIVE" value in the CARMEN_Results sheet
        red_cells = set()  # to store (row_idx, col_idx) for "POSITIVE" cells
    
        for df, sheet_name in zip(dataframes, sheet_names):
            # write each DataFrame to its respective sheet
            df.fillna('', inplace=True)
            df.to_excel(writer, sheet_name=sheet_name, index=True)
            
            # set col dimensions
            worksheet = writer.sheets[sheet_name]
            worksheet.set_column(0, 0, 20)
            for col_idx in range(1, df.shape[1]):  # Other columns
                worksheet.set_column(col_idx, col_idx, 16)
            
            # modify colors of output
            if sheet_name == "CARMEN_Hit_Results":
                worksheet = writer.sheets[sheet_name]
                # define font colors for POSITIVE and NEGATIVE
                red_font = writer.book.add_format({'font_color': 'F97609', 'bold': True})  # Red for POSITIVE - change to orange for color blind accessibiliy
                green_font = writer.book.add_format({'font_color': '1A85FF'})  # Green for NEGATIVE - change to blue for color blind accessibility
                black_font = writer.book.add_format({'font_color': '000000'})  # Black for other values

                # apply text color formatting
                for row_idx, row in enumerate(df.values, start=1):  
                    for col_idx, cell_value in enumerate(row, start=1):
                        cell_str = str(cell_value)
                        #print(f"Row {row_idx}, Col {col_idx}, Value: {cell_value}, Cell_str: {cell_str}")
                        if "NEGATIVE" in cell_str:
                            #print(f"Applying green font to {cell_value}")
                            worksheet.write(row_idx, col_idx, cell_value, green_font)
                        elif "POSITIVE" in cell_str:
                            #print(f"Applying red font to {cell_value}")
                            worksheet.write(row_idx, col_idx, cell_value, red_font)
                            red_cells.add((row_idx, col_idx))  # Track the red cells
                        else:
                            worksheet.write(row_idx, col_idx, cell_value, black_font)
            
            # apply red color to same cells in other sheets (NTC_Normalized_Quant_Results and Binary_Results)
            if sheet_name in ["NTC_Normalized_Quant_Results", "Binary_Results"]:
                worksheet = writer.sheets[sheet_name]
                for row_idx, row in enumerate(df.values, start=1):
                    for col_idx, cell_value in enumerate(row, start=1):
                        if (row_idx, col_idx) in red_cells:
                            worksheet.write(row_idx, col_idx, cell_value, red_font)  # Apply red formatting

    print(f"All FLAGGED FILES have been saved to {output_file_path}.")

except Exception as e:
    print(f"Error saving single Excel file: {e}. Results are saved as 5 individual csv files.")
    ### FILE 1: t13_hit_output as Results_Summary
    # convert the t13 hit output to an excel file with green/red conditional formatting for NEG/POS results
    t13_hit_output_file_path = os.path.join(res_subfolder, f'Results_Summary_{barcode_assignment}.xlsx')

    # create the Excel writer
    with pd.ExcelWriter(t13_hit_output_file_path, engine="xlsxwriter") as writer:
        # write the DataFrame to an Excel sheet
        fl_t13_hit_output.to_excel(writer, sheet_name="Sheet1", index=True)
        workbook = writer.book
        worksheet = writer.sheets["Sheet1"]

        # define font colors for POSITIVE and NEGATIVE
        red_font = writer.book.add_format({'font_color': 'FF0000', 'bold': True})  # Red for POSITIVE
        green_font = writer.book.add_format({'font_color': '008000'})  # Green for NEGATIVE

        # apply text color formatting
        for row_idx, row in enumerate(t13_hit_output.itertuples(index=False), start=1):  # Start from row 3 (after header and INVALID ASSAY label)
            for col_idx, cell_value in enumerate(row, start=1):
                cell_str = str(cell_value)
                #print(f"Row {row_idx}, Col {col_idx}, Value: {cell_value}, Cell_str: {cell_str}")
                if "NEGATIVE" in cell_str:
                    #print(f"Applying green font to {cell_value}")
                    worksheet.write(row_idx, col_idx, cell_value, green_font)
                elif "POSITIVE" in cell_str:
                    #print(f"Applying red font to {cell_value}")
                    worksheet.write(row_idx, col_idx, cell_value, red_font)
                    



    ### FILE 2: rounded_t13_quant_norm as NTC_Quant_Normalized_Results
    quant_output_ntcNorm_file_path = os.path.join(res_subfolder, f'NTC_Normalized_Quantitative_Results_Summary_{barcode_assignment}.csv')
    fl_rounded_t13_quant_norm.to_csv(quant_output_ntcNorm_file_path, index=True)

    ### File 3: summary_samples_df as Positives_Summary
    summary_pos_samples_file_path = os.path.join(res_subfolder, f'Positives_Summary_{barcode_assignment}.csv')
    fl_summary_samples_df.to_csv(summary_pos_samples_file_path, index=True)

    ### File 4: ntc_thresholds_output as NTC_Thresholds
    ntc_thresholds_output_file_path = os.path.join(res_subfolder, f'NTC_thresholds_{barcode_assignment}.csv')
    fl_rounded_ntc_thresholds_output.to_csv(ntc_thresholds_output_file_path, index=True)

    ### File 5: t13_hit_binary_output as t13_hit_Binary 
    t13_hit_binary_output_file_path = os.path.join(rd_subfolder, f't13__{barcode_assignment}_hit_binary.csv')
    fl_t13_hit_binary_output.to_csv(t13_hit_binary_output_file_path, index=True)
        
 
######################################################################################################################################################   
# instantiate Plotter from plotting.py
heatmap_generator = Plotter()

tgap = 3 # time gap between mixing of reagents (end of chip loading) and t0 image in minutes
# tp = list of timepoints (t1, t2, etc)
#unique_crRNA_assays = list(set(crRNA_assays))
unique_crRNA_assays = list(OrderedDict.fromkeys(crRNA_assays))
heatmap = heatmap_generator.plt_heatmap(tgap, barcode_assignment,final_med_frames, samples_list, unique_crRNA_assays, timepoints)

# Make subfolder in the output folder in your path's wd if it hasn't been made already
heatmaps_subfolder = os.path.join(rd_subfolder, f'Heatmaps_by_Timepoint_{barcode_assignment}')
if not os.path.exists(heatmaps_subfolder):
    os.makedirs(heatmaps_subfolder)

# save heatmap per timepoint
for i, t in enumerate(timepoints, start=1):
    #csv = convert_df(final_med_frames[t])
    heatmap_filename = os.path.join(heatmaps_subfolder, f'Heatmap_t{i}_{barcode_assignment}.png')
    fig = heatmap[t].savefig(heatmap_filename, bbox_inches = 'tight', dpi=80)
    plt.close(fig)

print(f"The heatmap plots saved to the folder, {heatmaps_subfolder} in {rd_subfolder} in {output_folder}.")

######################################################################################################################################################   
# instantiate t13_Plotter from t13_plotting.py
t13_heatmap_generator = t13_Plotter()

# Plot the t13 heatmap with annotations
heatmap_t13_quant_norm = t13_heatmap_generator.t13_plt_heatmap(tgap, barcode_assignment,t13_quant_norm, samples_list, unique_crRNA_assays, timepoints, invalid_samples, invalid_assays, rnasep_df)
heatmap_t13_quant_norm_filename = os.path.join(res_subfolder, f'NTC_Normalized_Heatmap_{barcode_assignment}.png')
fig = heatmap_t13_quant_norm.savefig(heatmap_t13_quant_norm_filename, bbox_inches = 'tight', dpi=80)
plt.close(fig)


######################################################################################################################################################   
# RedCap Integration
# set it as you have to enter a CLI arg for Redcap to run this code

if len(CLI_arg) > 2 and CLI_arg[2] == 'REDCAP':
    # instantiate RedCapper from flags.py
    redcapper = RedCapper()

    # make copy of binary output file from RESULTS Excel sheet
    fl_t13_hit_binary_output_2 = fl_t13_hit_binary_output.copy()

    # apply redcapper to fl_t13_hit_binary_output_2 df
    threshold = CLI_arg[1]
    redcap_t13_hit_binary_output, samplesDF  = redcapper.build_redcap(fl_t13_hit_binary_output_2, date, barcode_assignment,threshold, software_version)
    
    # save REDCAP file
    redcap_t13_hit_binary_output_file_path = os.path.join(res_subfolder, f'REDCAP_{barcode_assignment}_{CLI_arg[1]}.csv')
    redcap_t13_hit_binary_output.to_csv(redcap_t13_hit_binary_output_file_path, index=False)
    print("REDCAP file generated.")
    print("Operation complete.")

else:
    print("User did not specify REDCAP as a command line argument, and it follows that the REDCAP file was not generated.")
    print("Operation complete.")


