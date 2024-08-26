import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class ntcContaminationChecker:
    def __init__(self):
        pass

    def ntc_cont(self, assigned_sig_norm):
        
        # filter sample for anything that contains NTC (case-insensitive)
        ntc_filtered_df = assigned_sig_norm[assigned_sig_norm['sample'].str.contains('NTC', case=False, na=False)]
        # filter all other samples and save it in a new df 
        all_else_filtered_df = assigned_sig_norm[~assigned_sig_norm['sample'].str.contains('NTC', case=False, na=False)]
        # initiate list of ntc_assay_dfs
        ntc_assay_dfs = [] 
        # check per assay_df 
        for assay in ntc_filtered_df['assay'].unique():
            # filter the df for the current assay
            assay_df = ntc_filtered_df[ntc_filtered_df['assay'] == assay]

            # Check if t13 value is greater than 0.5 for any sample
            for index, row in assay_df.iterrows():
                if row['t13'] > 0.5:
                    # Check if there is still more than one row left in assay_df
                    if len(assay_df) > 1:
                        # Remove the row where t13 is greater than 0.5
                        assay_df = assay_df.drop(index)
                    else:
                        # If there is only one row left, set row['t13'] to 0.5
                        assay_df.at[index, 't13'] = 0.5
            # add assay_df back to list of ntc_assay_dfs
            ntc_assay_dfs.append(assay_df)
        
        # convert ntc_assay_dfs list into a df 
        combined_ntc_assay_dfs = pd.concat(ntc_assay_dfs, ignore_index=True)

        assigned_sig_norm = pd.concat([combined_ntc_assay_dfs, all_else_filtered_df], ignore_index=True)

        # pull the samples list 
            
        return assigned_sig_norm