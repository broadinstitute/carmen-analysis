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
# normalize the signal 

processor = DataProcessor()
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

median = MedianSort(crRNA_assays)
final_med_frames = median.create_median(assigned_norms['signal_norm_raw'])

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
t13_dataframe_copy1 = pd.DataFrame(t13_dataframe_orig)
t13_dataframe_copy2 = pd.DataFrame(t13_dataframe_orig)

# at this point, we have created a t1 thru t13 dataframe and exported all these dataframes as csv files in our output folder
# now we need to threshold the t13 csv and mark signals >= threshold as positive and < threshold as negative

# instantiate Thresholder from threshold.py
thresholdr = Thresholder()

# apply the NTC thresholding to the t13_dataframe to produce a new dataframe with the positive/negative denotation
# and save the file to your working directory
ntc_thresholds_output, t13_hit_output = thresholdr.raw_thresholder(t13_dataframe_copy1)

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

# instantiate Summarized from summary.py
# apply summarizer to the t13_dataframe to produce a new dataframe tabulating all of the positive samples
summary = Summarized()
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


