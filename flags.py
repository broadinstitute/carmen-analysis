import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class Flagger:
    def __init__(self):
        pass
    
    # method
    def assign_flags(self, QC_score_per_assay_df, t13_hit, t13_quant_norm, pos_samples_df, ntc_thresh, t13_hit_binary):
    
        files = [t13_hit, t13_quant_norm, pos_samples_df, ntc_thresh, t13_hit_binary]
        flagged_files = [] # store modified files after applying flags

        ### CPC flags 
        ## need to be added to t13_hit_output, rounded_t13_quant_norm, summary_samples_df, ntc_thresholds_output, t13_hit_binary_output
        for file in files: 
            flagged_file = file.copy() # work on a copy of the orig file
            invalid_assays = []  #  track which assays are invalid based on QC3 test results
            for row in QC_score_per_assay_df.itertuples():
                if row.Index == 'QC3: CPC':
                    for assay in QC_score_per_assay_df.columns: 
                        score = getattr(row, assay)
                        if score == 0: # CPC test has failed, the assay is invalid
                            invalid_assays.append(assay) # add to invalid assays list
            
            if invalid_assays: # if there are invalid assays
                # add asterisk to assay name in column heading of all files
                flagged_file.columns = [f'{col.upper()}*' if col.lower() in [assay.lower() for assay in invalid_assays] else col.upper() for col in flagged_file.columns]
                
                # add INVALID ASSAY below the assay name in column heading of all files
                invalid_row = []
                for col in flagged_file.columns:
                    if col.rstrip('*').lower() in [assay.lower() for assay in invalid_assays]:
                        invalid_row.append('INVALID ASSAY')  # mark invalid assays with this label
                    else:
                        invalid_row.append('') # this way, invalid_row has same dimensions as flagged_file's cols
                invalid_row_df = pd.DataFrame([invalid_row], columns=flagged_file.columns)
                header = flagged_file.iloc[:0] # split flagged file into header and rest of the data
                rest_of_data = flagged_file.iloc[1:]  
                flagged_file = pd.concat([header, invalid_row_df, rest_of_data], ignore_index=True) # concatenate all
                
                # if there are invalid assays marked with *, add a legend at the bottom of the file
                label = '*: This assay is considered invalid due to failing Quality Control Test #3, which evaluates performance of the Combined Positive Control sample.'
                invalid_legend_label = pd.DataFrame(data=[[label] + [pd.NA]*(len(flagged_file.columns) - 1)], columns=flagged_file.columns)
                invalid_legend_label_filled = invalid_legend_label.fillna('')
                # concatenate the invalid_legend label df to file df 
                flagged_file = pd.concat([flagged_file, invalid_legend_label_filled], ignore_index=True) # concatenate

            flagged_files.append(flagged_file) # add flagged file to the list
                
        ### NTC flags
        ## need to be added 

        ### no-crRNA flags


        ### RNaseP flags
        for 

        # build a large for loop that iterates over all these files?

        ### FILE 1: t13_hit_output as Results_Summary
        ### FILE 2: rounded_t13_quant_norm as NTC_Quant_Normalized_Results
        ### File 3: summary_samples_df as Positives_Summary
        ### File 4: ntc_thresholds_output as NTC_Thresholds
        ### File 5: t13_hit_binary_output as t13_hit_Binary 

        #return file, invalid_legend_label_filled, invalid_legend_label, invalid_assays
        return invalid_row, flagged_files