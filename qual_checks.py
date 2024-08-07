import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class Qual_Ctrl_Checks:
    def __init__(self):
        pass

    def quality_ctrl(self, binary_t13_df):
        binary_t13_df.loc['']

        return

    def ndc_check(self, binary_t13_df):
        # filter the rows to find the NDC
        ndc_rows = binary_t13_df[binary_t13_df.index.str.contains('NDC')]

        # for the rows containing NDC, collect the (row name, column name) for cells that are positive
        positive_ndc = []
        for row_name, row in ndc_rows.iterrows():
            positive_ndc_assays = []
            for col_name, cell_value in row.items():
                if 'positive' in str(cell_value).lower():
                    positive_ndc_assays.append(col_name)
            if positive_ndc_assays:
                positive_ndc.append((row_name, positive_ndc_assays))
        return positive_ndc
    
    
    def cpc_check(self, binary_t13_df):
        # filter the rows to find the CPC
        cpc_rows = binary_t13_df[binary_t13_df.index.str.contains('CPC')]

        # for the rows containing CPC, collect the (row name, column name) for cells that are negative
        negative_cpc = []
        for row_name, row in cpc_rows.iterrows():
            negative_cpc_assays = []
            for col_name, cell_value in row.items():
                if 'negative' in str(cell_value).lower():
                    negative_cpc_assays.append(col_name)
            if negative_cpc_assays:
                negative_cpc.append((row_name, negative_cpc_assays))
        return negative_cpc 
    
    def rnasep_check(self, binary_t13_df):
        # lowercase all column names and filter the col to find rnasep col
        binary_t13_df.columns = binary_t13_df.columns.str.lower()
        rnasep_col = binary_t13_df.filter(like='rnasep', axis=1)

        # for the rows containing CPC, collect the (row name, column name) for cells that are negative
        negative_rnasep = []
        for row_name, row in rnasep_col.iterrows():
            negative_rnasep_samples = []
            for col_name, cell_value in row.items():
                if 'negative' in str(cell_value).lower():
                    negative_rnasep_samples.append(row_name)
            if negative_rnasep_samples:
                negative_rnasep.append((col_name, negative_rnasep_samples))
        return rnasep_col, negative_rnasep 
    
    def ntc_check(self, assigned_sig_norm):
        # filter for columns t13, assay, and sample
        filtered_df = assigned_sig_norm[['t13', 'assay', 'sample']]

        # filter sample for anything that contains NTC (case-insensitive)
        ntc_filtered_df = filtered_df[filtered_df['sample'].str.contains('NTC', case=False, na=False)]

        high_raw_ntc = []

        for assay in ntc_filtered_df['assay'].unique():
            # filter the df for the current assay
            assay_df = ntc_filtered_df[ntc_filtered_df['assay'] == assay]

            # Check if t13 value is greater than 0.5 for any sample
        for _, row in assay_df.iterrows():
            if row['t13'] > 0.5:
                # Collect the sample name, assay, and t13 signal
                high_raw_ntc.append((row['sample'], row['assay'], row['t13']))

        return high_raw_ntc

    

            

          