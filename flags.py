import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class Flagger:
    def __init__(self):
        pass
    
    # method
    def assign_flags(self, high_raw_ntc_signal_df, rnasep_df, QC_score_per_assay_df, t13_hit, t13_quant_norm, pos_samples_df, ntc_thresh, t13_hit_binary):
    
        files = [t13_hit, t13_quant_norm, pos_samples_df, ntc_thresh, t13_hit_binary] # 0, 1, 2, 3, 4
        flagged_files = [] # store modified files after applying flags

        ### CPC flags 
        ## need to be added to t13_hit_output, rounded_t13_quant_norm, summary_samples_df, ntc_thresholds_output, t13_hit_binary_output
        for i, file in enumerate(files): 
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
                invalid_row_df.index = ["Assay Valid?"]
                data = flagged_file.iloc[0:]  
                flagged_file = pd.concat([invalid_row_df, data], ignore_index=False) # concatenate all
                
                # if there are invalid assays marked with *, add a legend at the bottom of the file
                label = 'This assay is considered invalid due to failing Quality Control Test #3, which evaluates performance of the Combined Positive Control sample.'
                invalid_legend_label = pd.DataFrame(data=[[label] + [pd.NA]*(len(flagged_file.columns) - 1)], columns=flagged_file.columns, index=["Legend for *:"])
                invalid_legend_label_filled = invalid_legend_label.fillna('')
                # concatenate the invalid_legend label df to file df 
                flagged_file = pd.concat([flagged_file, invalid_legend_label_filled], ignore_index=False) # concatenate

            
            ### NTC flags
            ## dagger flag needs to be added to t13_hit_output, rounded_t13_quant_norm, summary_samples_df, t13_hit_binary_output
            if i in {0,1,2,4}:
                processed_samples = set()
                for _, row in high_raw_ntc_signal_df.iterrows():
                    for col in high_raw_ntc_signal_df.columns: # cols are Sample, Assay, t13 
                        cont_ntc_sample = row['Sample'] # NEG NTC sample
                        cont_ntc_assay = row['Assay'] # NTC assay
                        # now iterate over the flagged file
                        for idx, sample_row in flagged_file.iterrows(): 
                            if cont_ntc_sample == idx:
                                # add † to each cell value
                                for assay_col in flagged_file.columns:  
                                    if assay_col.upper() == cont_ntc_assay.upper():
                                        # check that the sample-assay pair has alr been processed
                                        if (cont_ntc_sample, cont_ntc_assay) not in processed_samples:
                                            processed_samples.add((cont_ntc_sample, cont_ntc_assay))
                                            # check if the value is NA (NaN)
                                            if pd.isna(sample_row[assay_col]):
                                                flagged_file.at[idx, assay_col] = '†'  # only dagger if value is NA
                                            else:
                                                flagged_file.at[idx, assay_col] = f"{sample_row[assay_col]}†"  # add dagger to the value
                         
                for _, row in high_raw_ntc_signal_df.iterrows(): 
                    for col in high_raw_ntc_signal_df.columns: 
                        cont_ntc_sample = row['Sample'] # NEG NTC sample
                        cont_ntc_assay = row['Assay'] # NTC assay
                        # now iterate over the flagged file
                        for idx, sample_row in flagged_file.iterrows(): 
                            if cont_ntc_sample == idx: 
                                # add ** to sample name
                                new_index = f"{idx}†"
                                flagged_file = flagged_file.rename(index={idx: new_index})
                       
                # add legend at the bottom of the file
                legend_added = False
                for index, sample_row in flagged_file.iterrows():
                    if '†' in str(index):
                        label = 'The NTC sample for this assay was removed from the analysis due to potential contamination.'
                        cont_NTC_legend_label = pd.DataFrame(data=[[label] + [pd.NA]*(len(flagged_file.columns) - 1)], columns=flagged_file.columns, index=["Legend for †:"])
                        cont_NTC_legend_label_filled = cont_NTC_legend_label.fillna('')
                        # concatenate the invalid_legend label df to file df 
                        flagged_file = pd.concat([flagged_file, cont_NTC_legend_label_filled], ignore_index=False) # concatenate
                        legend_added = True
                        break
                
            ## explanation needs to be added to ntc_thresholds_output
            #if i == 3:


            ### no-crRNA flags

            
            ### RNaseP flags 
            ## need to be added to t13_hit_output, rounded_t13_quant_norm, t13_hit_binary_output
            if i in {0, 1, 4} :  # modify only specific files
                # add asterisk to negative rnasep samples
                for _, row in rnasep_df.iterrows(): 
                    for col in rnasep_df.columns: # col is the RNaseP assay
                        rnasep_sample = row[col] # NEG RNaseP sample
                        # now iterate over the flagged file
                        for index, sample_row in flagged_file.iterrows():     
                            if rnasep_sample == index: 
                                # add ** to each cell value
                                for assay_col in flagged_file.columns:  
                                    if col.upper() == assay_col:
                                        flagged_file.at[index, assay_col] = f"{sample_row[assay_col]}**" 
                
                for _, row in rnasep_df.iterrows(): 
                    for col in rnasep_df.columns: # col is the RNaseP assay
                        rnasep_sample = row[col] # NEG RNaseP sample
                        # now iterate over the flagged file
                        for index, sample_row in flagged_file.iterrows():     
                            if rnasep_sample == index: 
                                # add ** to sample name
                                new_index = f"{index}**"
                                flagged_file = flagged_file.rename(index={index: new_index})
                            
                # add legend at the bottom of the file
                legend_added = False
                for index, sample_row in flagged_file.iterrows():
                    if '**' in index:
                        label = 'This sample is negative for human internal control, RNaseP. There are a few different implications of this result. See Quality Control Report for further explanation.'
                        neg_rnasep_legend_label = pd.DataFrame(data=[[label] + [pd.NA]*(len(flagged_file.columns) - 1)], columns=flagged_file.columns, index=["Legend for **:"])
                        neg_rnasep_legend_label_filled = neg_rnasep_legend_label.fillna('')
                        # concatenate the invalid_legend label df to file df 
                        flagged_file = pd.concat([flagged_file, neg_rnasep_legend_label_filled], ignore_index=False) # concatenate
                        legend_added = True
                        break
                
            flagged_files.append(flagged_file) # add flagged file to the list          

        return flagged_files