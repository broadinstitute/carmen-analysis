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
from reader import DataReader
from norm import DataProcessor
from matcher import DataMatcher
from median_frame import MedianSort
from threshold import Thresholder

def convert_df(df):
    return df.to_csv(encoding='utf-8')

path = Path(os.getcwd())
path_dir = path
all_files = set(map(os.path.basename, path_dir.glob('*')))
assignment_files = sorted([fname for fname in all_files if (fname.endswith(".xlsx"))])
data_files = sorted([fname for fname in all_files if (fname.endswith(".csv"))])

# instantiate the DataReader from reader.py
# first load in the data from the assignment sheet 
if assignment_files:
    file_path = assignment_files[0] # assignment_files contains file paths
    # file_like_object = BytesIO(assignment_files[0].read())
    assignment_df = pd.read_excel(file_path)
    print(f"This is the Assignment Sheet that was loaded: {assignment_files[0].name}")     
    
    assignment_file_name = assignment_files[0].name

    match = re.match(r"^(\d+)_.*", assignment_file_name)
    barcode_assignment = ""
    if match:
        s= match.group(1)
        barcode_assignment = match.group(1)
    print(f"IFC barcode: {barcode_assignment}")
else:
    print("Please upload an Assignment Sheet.")

# then load in the data file 
if data_files:
    file_like_object = BytesIO(data_files[0].read())
    df_data = pd.read_csv(file_like_object, on_bad_lines="skip")
    print(f"This is the Data File that was loaded: {data_files[0].name}")
   
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
else:
    print("Please upload a Data File.")

# at this point, we have loaded the assignment sheet and have sorted through the loaded data file to create a dict of dataframes 

# instantiate DataProcessor from norm.py
# normalize the signal 
if data_files is not None and assignment_files is not None:
    processor = DataProcessor()
    normalized_dataframes = processor.background_processing(dataframes)
    
    # make an output folder in your path's wd if it hasn't been made already
    output_folder = 'output'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # the output of DataProcessor is an array of dataframes where 1st is signal_norm and 2nd is ref_norm
    # save these outputs to your output folder
    normalized_dataframes[0].to_csv(os.path.join(output_folder, 'signal_norm.csv'), index=False)
    normalized_dataframes[1].to_csv(os.path.join(output_folder, 'ref_norm.csv'), index=False)

    # instantiate DataMatcher from matcher.py
    matcher = DataMatcher() 
    assigned_norms, assigned_lists = matcher.assign_assays(assignment_files[0], normalized_dataframes['ref_norm'], normalized_dataframes['signal_norm'])

    # save the output of assigned_norms array to your output folder
    # 1st item is the assigned signal_norm_raw csv and 2nd is the assigned ref_norm_csv
    assigned_norms[0].to_csv(os.path.join(output_folder, 'assigned_signal_norm.csv'), index=False)
    assigned_norms[1].to_csv(os.path.join(output_folder, 'assigned_ref_norm.csv'), index=False)

    
    crRNA_assays = pd.read_excel(assignment_files[0],sheet_name='assays')
    crRNA_array = crRNA_assays.values.flatten()

    median = MedianSort(crRNA_list = crRNA_array)
    final_med_frames = median.create_median(assigned_norms['signal_norm_raw'])

    timepoints = list(final_med_frames.keys())
    for i, t in enumerate(timepoints, start=0):
        csv = convert_df(final_med_frames[t])
        filename = os.path.join(output_folder, f't{i}_{barcode_assignment}.csv')
        csv.to_csv(filename, index=False)

    # since we want to explicitly manipulate t13_csv, it is helpful to have the t13 df referenced outside of the for loop
    last_key = list(final_med_frames.keys())[-1]
    t13_dataframe = final_med_frames[last_key]

    # at this point, we have created a t1 thru t13 dataframe and exported all these dataframes as csv files in our output folder
    # now we need to threshold the t13 csv and mark signals >= threshold as positive and < threshold as negative

    # instantiate Thresholder from threshold.py
    thresholdr = Thresholder()

    # apply the NTC thresholding to the t13_dataframe to produce a new dataframe with the positive/negative denotation
    # and save the file to your working directory
    t13_hit_output = thresholdr.raw_thresholder(t13_dataframe)
    hit_output_file_path = os.path.join(output_folder, f't13_{barcode_assignment}_hit_output.csv')
    t13_hit_output.to_csv(hit_output_file_path, index=False)

   
else:
    print("Please upload an Assignment Sheet and a Data File.")





