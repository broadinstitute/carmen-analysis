#dataloading assets
import sys
import pandas as pd
import numpy as np 
import math 
#plotting imports
import matplotlib.pyplot as plt 
import seaborn as sns 
#file imports
from io import BytesIO
import base64
from pathlib import Path
import os 
from os import path
import glob 
import re
from collections import OrderedDict
from reader import DataReader
from norm import DataProcessor
from matcher import DataMatcher
from median_frame import MedianSort
from threshold import Thresholder
from ntcnorm import Normalized
from summary import Summarized
from plotting import Plotter
from tqdm import tqdm
from qual_checks import Qual_Ctrl_Checks
from binary_results import Binary_Converter
from ntc_con_check import ntcContaminationChecker
from assay_qc_score import Assay_QC_Score


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
   
    reader = DataReader() # make am empty class object that corresponds to the DataReader() object from reader.py

    phrases_to_find = [
        "Raw Data for Passive Reference ROX",
        "Raw Data for Probe FAM-MGB",
        "Bkgd Data for Passive Reference ROX",
        "Bkgd Data for Probe FAM-MGB"
    ]
    file_like_object.seek(0)
    
    # Extract dataframes from each CSV file
    dataframes = reader.extract_dataframes_from_csv(file_like_object, phrases_to_find)

# at this point, we have loaded the assignment sheet and have sorted through the loaded data file to create a dict of dataframes 

# instantiate DataProcessor from norm.py
processor = DataProcessor()
# normalize the signal 
normalized_dataframes = processor.background_processing(dataframes)

# make an output folder in your path's wd if it hasn't been made already
output_folder = f'output_{barcode_assignment}'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# the output of DataProcessor is an array of dataframes where 1st is signal_norm and 2nd is ref_norm
# save these outputs to your output folder
normalized_dataframes['signal_norm'].to_csv(os.path.join(output_folder, 'signal_norm.csv'), index=True)
normalized_dataframes['ref_norm'].to_csv(os.path.join(output_folder, 'ref_norm.csv'), index=True)

# instantiate DataMatcher from matcher.py
matcher = DataMatcher() 
assigned_norms, assigned_lists = matcher.assign_assays(assignment_files[0], normalized_dataframes['ref_norm'], normalized_dataframes['signal_norm'])

# save the output of assigned_norms array to your output folder
# 1st item is the assigned signal_norm_raw csv and 2nd is the assigned ref_norm_csv
assigned_norms['signal_norm_raw'].to_csv(os.path.join(output_folder, 'assigned_signal_norm.csv'), index=True)
assigned_norms['ref_norm_raw'].to_csv(os.path.join(output_folder, 'assigned_ref_norm.csv'), index=True)

samples_list = assigned_lists['samples_list']
crRNA_assays = assigned_lists['assay_list']

# instantiate ntcContaminationChecker from ntc_con_check.py
ntcCheck = ntcContaminationChecker()

# make a copy of assigned_signal_norm
assigned_signal_norm = pd.DataFrame(assigned_norms['signal_norm_raw']).copy() # make a copy of assigned_signal_norm dataframe
# create df of filtered assigned_signal_norm by applying the NTC check to remove any NTCs whose raw signal suggests contamination
assigned_signal_norm_with_NTC_check = ntcCheck.ntc_cont(assigned_signal_norm) # feed this into MedianSort

# instantiate MedianSort from median_frame.py
median = MedianSort(crRNA_assays)
final_med_frames = median.create_median(assigned_signal_norm_with_NTC_check)

# Output needs to be rounded to 4 digits
rounded_final_med_frames = {}
# Define the number of decimals for rounding
decimals = 5

# Iterate through each row and column
for key, df in final_med_frames.items():
    rounded_df = pd.DataFrame(index=df.index, columns=df.columns)
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            # Round each value to the specified number of decimals
            rounded_df.iloc[i, j] = round(df.iloc[i, j], decimals)
    rounded_final_med_frames[key] = rounded_df

timepoints = list(rounded_final_med_frames.keys())
for i, t in enumerate(timepoints, start=1):
    filename = os.path.join(output_folder, f't{i}_{barcode_assignment}.csv')
    csv = rounded_final_med_frames[t].to_csv(filename, index=True)

# since we want to explicitly manipulate t13_csv, it is helpful to have the t13 df referenced outside of the for loop
last_key = list(rounded_final_med_frames.keys())[-1]
t13_dataframe_orig = rounded_final_med_frames[last_key]
t13_dataframe_copy1 = pd.DataFrame(t13_dataframe_orig).copy()
t13_dataframe_copy2 = pd.DataFrame(t13_dataframe_orig).copy()

# at this point, we have created a t1 thru t13 dataframe and exported all these dataframes as csv files in our output folder
# now we need to threshold the t13 csv and mark signals >= threshold as positive and < threshold as negative

# The premise of the code is that different viral panels require different thresholding
# So the user will specifiy command line arguments as per the ReadMe instructions
# and this portion of the code is meant to access the CI arguments and modify the threshold specified in the code

CLI_arg = sys.argv

# instantiate Thresholder from threshold.py
thresholdr = Thresholder()
unique_crRNA_assays = list(set(crRNA_assays))

# apply the NTC thresholding to the t13_dataframe to produce a new dataframe with the positive/negative denotation
# and save the file to your working directory
ntc_PerAssay, ntc_thresholds_output, t13_hit_output = thresholdr.raw_thresholder(unique_crRNA_assays, assigned_norms['signal_norm_raw'], t13_dataframe_copy1, CLI_arg[1])


# make copies of t13_hit_output csv for downstream summaries and quality control checks
t13_hit_output_copy1 = pd.DataFrame(t13_hit_output).copy() # make a copy of t13_hit_output # used in ndc qual check
t13_hit_output_copy2 = pd.DataFrame(t13_hit_output).copy() # make a copy of t13_hit_output # used in cpc qual check
t13_hit_output_copy3 = pd.DataFrame(t13_hit_output).copy() # make a copy of t13_hit_output # used in rnasep qual check
t13_hit_output_copy4 = pd.DataFrame(t13_hit_output).copy() # make a copy of t13_hit_output # used in t13_hit_binary_output generation
t13_hit_output_copy5 = pd.DataFrame(t13_hit_output).copy() # make a copy of t13_hit_output
t13_hit_output_copy6 = pd.DataFrame(t13_hit_output).copy() # make a copy of t13_hit_output

ntc_thresholds_output_file_path = os.path.join(output_folder, f'NTC_thresholds_{barcode_assignment}.csv')
hit_output_file_path = os.path.join(output_folder, f't13_{barcode_assignment}_hit_output.csv')

ntc_thresholds_output.to_csv(ntc_thresholds_output_file_path, index=True)
t13_hit_output.to_csv(hit_output_file_path, index=True)

 # instantiate NTC_Normalized from ntcnorm.py
ntcNorm = Normalized()
 # apply ntc_normalizr to the t13_dataframe to produce a new dataframe with all values divided by the mean NTC for that assay
t13_quant_hit_norm = ntcNorm.normalizr(t13_dataframe_copy2)
quant_hit_output_ntcNorm_file_path = os.path.join(output_folder, f't13_{barcode_assignment}_quant_ntcNorm.csv')
t13_quant_hit_norm.to_csv(quant_hit_output_ntcNorm_file_path, index=True)

# instantiate Binary_Converter from binary_results.py
binary_num_converter = Binary_Converter()
# apply hit_numeric_conv to the the t13_hit_output to produce a new dataframe with all pos/neg converted to binary 1/0 output
t13_hit_binary_output = binary_num_converter.hit_numeric_conv(t13_hit_output_copy4)
t13_hit_binary_output_file_path = os.path.join(output_folder, f't13_{barcode_assignment}_hit_binary_output.csv')
t13_hit_binary_output.to_csv(t13_hit_binary_output_file_path, index=True)

# make copies of t13_hit_binary_output for downstream utilization in coinf check and assay level eval
t13_hit_binary_output_copy1 = pd.DataFrame(t13_hit_binary_output).copy() # used in coninf check
t13_hit_binary_output_copy2 = pd.DataFrame(t13_hit_binary_output).copy() 

# instantiate Summarized from summary.py
summary = Summarized()
# apply summarizer to the t13_dataframe to produce a new dataframe tabulating all of the positive samples
summary_samples_df = summary.summarizer(t13_hit_output)
summary_pos_samples_file_path = os.path.join(output_folder, f'Positives_Summary_{barcode_assignment}.csv')
summary_samples_df.to_csv(summary_pos_samples_file_path, index=True)


# instantiate Plotter from plotting.py
heatmap_generator = Plotter()

tgap = 3 # time gap between mixing of reagents (end of chip loading) and t0 image in minutes
# tp = list of timepoints (t1, t2, etc)
#unique_crRNA_assays = list(set(crRNA_assays))
unique_crRNA_assays = list(OrderedDict.fromkeys(crRNA_assays))
heatmap = heatmap_generator.plt_heatmap(tgap, barcode_assignment,final_med_frames, samples_list, unique_crRNA_assays, timepoints)

# save heatmap per timepoint
for i, t in enumerate(timepoints, start=1):
    #csv = convert_df(final_med_frames[t])
    heatmap_filename = os.path.join(output_folder, f'Heatmap_t{i}_{barcode_assignment}.png')
    fig = heatmap[t].savefig(heatmap_filename, bbox_inches = 'tight', dpi=80)
    plt.close(fig)

print(f"The heatmap plots saved to the folder, {output_folder}")

# instantiate Assay_QC_Score
assayScorer = Assay_QC_Score()
# take in t13_hit_binary_output as the df to build off of 
QC_score_per_assay_df = assayScorer.assay_level_score(t13_hit_binary_output)
assay_lvl_QC_score_file_path = os.path.join(output_folder, f'Assay_Level_QC_Metrics_{barcode_assignment}.csv')
QC_score_per_assay_df.to_csv(assay_lvl_QC_score_file_path, index=True)

# write text file explaining the QC score
assayScores_Explanation = []

assayScores_Explanation.append(f"The csv file, Assay_Level_QC_Metrics_{barcode_assignment}.csv, scores each viral assay in this experiment against four quality control (QC) tests.\n")
assayScores_Explanation.append("Quality Control (QC) Test #1: Assay Performance Evaluation Based on NTC Sample Results\n")
assayScores_Explanation.append(f"The first QC test assesses the performance of each assay by examining the percentage of negative results among the NTC (No Target Control) samples.")
assayScores_Explanation.append(f"The result from QC Test #1 is a score per assay. Below are the possible QC Scores that an assay can earn based on this evaluation:")
assayScores_Explanation.append(f"   - QC Score = 0: If <=25% of NTC controls are negative for an assay, the assay has failed this test.")
assayScores_Explanation.append(f"   - QC Score = 0.25: If the percent of negative NTC controls for an assay is >25% and <=50%, the assay has marginally passed this test.")          
assayScores_Explanation.append(f"   - QC Score = 0.50: If the percent of negative NTC controls for an assay is >50% and <=75%, the assay has partially passed this test.")            
assayScores_Explanation.append(f"   - QC Score = 0.75: If the percent of negative NTC controls for an assay is >75% and <100%, the assay has nearly passed this test.")         
assayScores_Explanation.append(f"   - QC Score = 1: If 100% of NTC controls are negative for an assay, the assay has completely passed this test.")    
assayScores_Explanation.append(f"The ideal outcome is for ALL viral assays tested to score 1 for QC Test #1.\n\n")     
 
assayScores_Explanation.append("Quality Control (QC) Test #2: Assay Performance Evaluation Based on NDC Sample Results\n")
assayScores_Explanation.append(f"The second QC test assesses the performance of each assay by examining the percentage of negative results among the NDC (No Detection Control) samples.")                            
assayScores_Explanation.append(f"The result from QC Test #2 is a score per assay. Below are the possible QC Scores that an assay can earn based on this evaluation:")                               
assayScores_Explanation.append(f"   - QC Score = 0: If <=50% of NDC controls are negative for an assay, the assay has failed this test.")
assayScores_Explanation.append(f"   - QC Score = 1: If >50% of NDC controls are negative for an assay, the assay has passed this test.")
assayScores_Explanation.append(f"The ideal outcome is for ALL viral assays tested to score 1 for QC Test #2.\n\n")     

assayScores_Explanation.append("Quality Control (QC) Test #3: Assay Performance Evaluation Based on CPC Sample Results\n")
assayScores_Explanation.append(f"The third QC test assesses the performance of each assay by examining the percentage of positive results among the CPC (Combined Positive Control) samples.")                            
assayScores_Explanation.append(f"The result from QC Test #3 is a score per assay. Below are the possible QC Scores that an assay can earn based on this evaluation:")                               
assayScores_Explanation.append(f"   - QC Score = 0: If <=50% of CPC controls are positive for an assay, the assay has failed this test.")
assayScores_Explanation.append(f"   - QC Score = 1: If >50% of CPC controls are negative for an assay, the assay has passed this test.")
assayScores_Explanation.append(f"The ideal outcome is for ALL viral assays tested, except for no-crRNA, to score 1 for QC Test #3.\n\n")  

assayScores_Explanation.append("Quality Control (QC) Test #4: Assay Performance Evaluation Based on Clinical Sample Results for RNaseP\n")
assayScores_Explanation.append(f"The fourth QC test assesses the performance of the RNaseP assay (included as a positive internal control for human samples) by examining the percentage of positive results among clinical samples.")                            
assayScores_Explanation.append(f"The result from QC Test #4 is a score for the RNaseP assay. Below is the explanation of the QC Score that the RNaseP assay can earn based on this evaluation:")                               
assayScores_Explanation.append(f"   - QC Score = 0.XX: The score is the fraction of clinical samples positive for RNaseP out of the total clinical samples tested.")
assayScores_Explanation.append(f"The ideal outcome is for the RNaseP assay to score 1 for QC Test #4.\n\n")  

assayScores_Explanation.append("The final score per assay is included for easy comparison of assay performance.\n")

# create and save an output text file containing the quality control checks
assayScores_file_path = os.path.join(output_folder, f'Assay_Performance_QC_Tests_{barcode_assignment}.txt')
with open(assayScores_file_path, 'w') as f:
    for line in assayScores_Explanation:
        f.write(line + '\n')

print(f"The four quality control tests to evaluate assay performance are complete. Their results have been saved to the folder, {output_folder}")

# instantiate Qual_Ctrl_Checks from qual-checks.py
qual_checks = Qual_Ctrl_Checks()

# initialize a list to collect all quality control checks
QC_lines = []

# apply ndc_check to the t13_hit_output df to generate a list of all ndc positive assays
ndc_positives = qual_checks.ndc_check(t13_hit_output_copy1)
QC_lines.append("1. Evaluation of No Detect Control (NDC) Contamination \n")
assay_list = []
if ndc_positives: 
    for ndc, assays in ndc_positives:
        assay_list.extend(assays)
    assay_str = ", ".join(assay_list)
    QC_lines.append(f"After thresholding against the NTC, {ndc} appears positive for the following assay(s): {assay_str}.")
    QC_lines.append("Please be advised to check the output files as well.\n\n")
else: 
    QC_lines.append("Since none of the NDCs ran in this experiment appear positive after thresholding against the NTC, we posit that there is likely no NDC contamination.")
    QC_lines.append("Please be advised to check the output files as well.\n\n")

# apply cpc_check to the t13_hit_output df to generate a list of all cpc negative assays
cpc_negatives = qual_checks.cpc_check(t13_hit_output_copy2)
QC_lines.append("2. Evaluation of Combined Positive Control (CPC) Validity \n")
assay_list = []
if cpc_negatives: 
    for cpc, assays in cpc_negatives:
        assay_list.extend(assays)
    assay_str = ", ".join(assay_list)
    QC_lines.append(f"After thresholding against the NTC, {cpc} appears negative for the following assay(s): {assay_str}.\n") 
    QC_lines.append("In the list provided above, if any of the CPCs show a negative result for an assay that is not the 'no-crRNA' negative control, then that particular assay should be considered invalid for this experiment.")
    QC_lines.append("Please be advised to check the output files as well.\n\n")
else: 
    QC_lines.append("Warning: First verify that your experiment included a CPC sample. If yes, proceed to the following CPC analysis.\n")
    QC_lines.append("After thresholding against the NTC, the CPC(s) appears as positive for all crRNA assays tested. It is expected for the CPC(s) to test as negative for 'no-crRNA' assay. There may be possible contamination of the 'no-crRNA' assay.")
    QC_lines.append("Please be advised to check the output files as well.\n\n")

# apply rnasep_check to the t13_hit_output df to generate a list of all rnasep negative samples
rnasep_df, rnasep_negatives = qual_checks.rnasep_check(t13_hit_output_copy3)
QC_lines.append("3. Evaluation of Human Samples for the Internal Control (RnaseP) \n")
rnasep_neg_samp_list = []
if rnasep_negatives: # there are some samples that are neg for RNAseP
    QC_lines.append("Warning: First verify that your experiment included a RNaseP assay. If yes, proceed to the following RNaseP analysis.\n")
    for rnasep, samples in rnasep_negatives:
        rnasep_neg_samp_list.extend(samples)
    rnasep_neg_samp_str = ", ".join(rnasep_neg_samp_list)
    QC_lines.append(f"After thresholding against the NTC, {rnasep} appears negative for the following sample(s): {rnasep_neg_samp_str}.\n")
    QC_lines.append("There are a few different reasons that a sample tests negative for RNaseP:\n")
    QC_lines.append("   (1) If the sample is negative for all assays (including RNaseP), then the most plausible hypothesis is that the viral extraction protocol used in this experiment needs to be examined. For optimal results, the extraction must be compatible with the Standard Operating Procedure (SOP) advised by the CARMEN team in the Sabeti Lab.")
    QC_lines.append("   ** Note: If the sample is negative for RNaseP and ALL other crRNA assays tested in this experiment, the sample should be rendered invalid.\n")
    QC_lines.append("   (2) If the sample is negative for RNaseP BUT positive for any other viral crRNA assay (excluding RNaseP or no-crRNA), then the most plausible hypothesis is that the sample’s viral titer may be too high compared to its RNaseP titer. This, thereby, renders the system possibly unable to detect RNaseP, leading to the sample testing negative for RNaseP.")
    QC_lines.append("   ** Note: If the sample is negative for RNaseP but positive for any other viral crRNA assay (excluding RNaseP or no-crRNA) tested in this experiment, the sample can still be included in the final results.\n")
    QC_lines.append("   (3) The source sample may have insufficient material, leading to a negative RNaseP signal and an invalid sample result.")
    QC_lines.append("Please be advised to check the output files as well.\n\n")
else: # all samples are positive for RNaseP - points to contamination
    QC_lines.append("The RNaseP internal control should test negative for the NTC and NDC negative control.")
    QC_lines.append("There are a few different reasons that all samples test positive for RNaseP. The most plausible hypothesis is that there is RNaseP contamination in this experiment. Precaution is advised to mitigate contamination avenues, especially at the RT-PCR (nucleic acid amplification) stage.")
    QC_lines.append("Please be advised to check the output files as well.\n\n")

# apply ntc_check to the t13_hit_output df to generate a list of all ntc positive assays
assigned_signal_norm_2 = pd.DataFrame(assigned_norms['signal_norm_raw']).copy() # make a copy of assigned_signal_norm dataframe


high_raw_ntc_signal = qual_checks.ntc_check(assigned_signal_norm_2)
QC_lines.append("4. Evaluation of No Target Control (NTC) Contamination \n")
if high_raw_ntc_signal:
    for sample, assay, t13 in high_raw_ntc_signal:
        QC_lines.append(f"The raw fluorescence signal for {sample} for {assay} is {t13}.\n") 
    QC_lines.append("Since the raw fluorescence signal for the listed sample(s) is above 0.5 a.u., it is being flagged to have a higher than normal signal for an NTC sample.")
    QC_lines.append("The range for typical raw fluorescence signal for an NTC sample is between 0.1 and 0.5 a.u. It is advised that the output files be examined further to evaluate potential NTC contamination.\n\n")
else:
    QC_lines.append("The raw fluorescence signal for each NTC sample across all crRNA assays tested in this experiment appears to be within the normal range of 0.1 and 0.5 a.u. Risk of NTC contamination is low.")
    QC_lines.append("Please be advised to check the output files as well.\n\n")

# apply coinfection check to t13_hit_binary_output to generate list of all samples that are positive for multiple assays
coinfection_df = qual_checks.coinf_check(t13_hit_binary_output_copy1)
QC_lines.append("5. Evaluation of Potential Co-Infected Samples\n")

coinfection_df_file_path = os.path.join(output_folder, f'Coinfection_Check_{barcode_assignment}.csv')
coinfection_df.to_csv(coinfection_df_file_path, index=True) # output needs to be csv of coninfection check

# in Qual_Check text file, add message saying "see coinf check csv" and "if any samples excpet CPC are flagged as being coinfected, there is risk of these samples being coinfected"
QC_lines.append("A preliminary evaluation for co-infection of a given sample against all tested assays has been completed.")
QC_lines.append(f"Please consult Codetection_Check_{barcode_assignment}.csv to see which samples may be potentially co-infected.\n")
QC_lines.append("   - If you have included Combined Positive Controls (CPCs) in this experiment, as recommended, these positive controls should be identified and listed among the flagged samples. CPCs are expected to show a “co-detection” with ALL of the assays being tested in this experiment.\n")
QC_lines.append("   - Samples are not flagged as “co-detected” based on positivity with RNaseP and a second assay. For a sample to be flagged during this Co-detection Check, it must test positive for at least two assays, excluding RNaseP.\n")
QC_lines.append("   - All other flagged samples should be further evaluated for potential co-infection.\n")
QC_lines.append("Please be advised to check the output files as well.")


# create and save an output text file containing the quality control checks
QCs_file_path = os.path.join(output_folder, f'Quality_Control_Flags_{barcode_assignment}.txt')
with open(QCs_file_path, 'w') as f:
    for line in QC_lines:
        f.write(line + '\n')

print(f"The quality control checks are complete and saved to the folder, {output_folder}")


