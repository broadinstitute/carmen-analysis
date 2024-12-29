import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import re

class Qual_Ctrl_Checks:
    def __init__(self):
        pass

    def ndc_check(self, binary_t13_df):
        # filter the rows to find the NDC
        ndc_rows = binary_t13_df[binary_t13_df.index.str.contains('NDC')]

        # for the rows containing NDC, collect the (row name, column name) for cells that are positive
        positive_ndc_dict = {}
        for row_name, row in ndc_rows.iterrows():
            positive_ndc_assays = []
            for col_name, cell_value in row.items():
                if 'positive' in str(cell_value).lower():
                    positive_ndc_assays.append(col_name)
            if positive_ndc_assays:
                positive_ndc_dict[row_name]= positive_ndc_assays
        # create a df to store the results
        positive_ndc_df = pd.DataFrame({col: pd.Series(values) for col, values in positive_ndc_dict.items()})
        
        return positive_ndc_df
    
    def cpc_check(self, binary_t13_df):  
        # filter the rows to find the CPC
        cpc_rows = binary_t13_df[binary_t13_df.index.str.contains('CPC')]
        assays = binary_t13_df.columns.tolist()
       
        if all(('_P1' in idx or '_P2' in idx or '_RVP' in idx) for idx in cpc_rows.index) and all(('_P1' in col or '_P2' in col or '_RVP' in col) for col in cpc_rows.columns):
            # filter cpc_rows further and stratify into cpc_rvp, cpc_p1, cpc_p2
            cpc_rvp_rows = cpc_rows[cpc_rows.index.str.contains('_RVP')]
            cpc_p1_rows = cpc_rows[cpc_rows.index.str.contains('_P1')]
            cpc_p2_rows = cpc_rows[cpc_rows.index.str.contains('_P2')]

            # initialize empty lists to store panel-specific assays
            rvp_assays = [] 
            p1_assays = []
            p2_assays = []

            # for the rows containing CPC per panel, collect the (row name, column name) for cells that are negative
            negative_cpc_dict = {}

            for assay in assays:
                if re.search(r'_RVP', assay): # divide assays in rvp, p1, and p2 assays based on label suffixes
                    rvp_assays.append(assay)
                if re.search(r'_P1', assay): # divide assays in rvp, p1, and p2 assays based on label suffixes
                    p1_assays.append(assay)
                if re.search(r'_P2', assay): # divide assays in rvp, p1, and p2 assays based on label suffixes
                    p2_assays.append(assay)
            
            negative_cpc_rvp_assays = []
            for rvp_assay in rvp_assays:
                for cpc_rvp_sample, row in cpc_rvp_rows.iterrows():
                    for col_name, cell_value in row.items():
                        if rvp_assay == col_name:
                            if 'negative' in str(cell_value).lower():
                                negative_cpc_rvp_assays.append(col_name)
                        if negative_cpc_rvp_assays:
                            negative_cpc_dict[cpc_rvp_sample] = negative_cpc_rvp_assays
            
            for cpc_p1_sample, row in cpc_p1_rows.iterrows():
                negative_cpc_p1_assays = []
                for col_name, cell_value in row.items():
                    for p1_assay in p1_assays:
                        if p1_assay == col_name:
                            if 'negative' in str(cell_value).lower():
                                negative_cpc_p1_assays.append(col_name)
                if negative_cpc_p1_assays:
                    negative_cpc_dict[cpc_p1_sample] = negative_cpc_p1_assays
           
            for cpc_p2_sample, row in cpc_p2_rows.iterrows():
                negative_cpc_p2_assays = []
                for col_name, cell_value in row.items():
                    for p2_assay in p2_assays:
                        if p2_assay == col_name:
                            if 'negative' in str(cell_value).lower():
                                negative_cpc_p2_assays.append(col_name)
                if negative_cpc_p2_assays:
                    negative_cpc_dict[cpc_p2_sample] = negative_cpc_p2_assays

            # create a df to store the results
            negative_cpc_df = pd.DataFrame({col: pd.Series(values) for col, values in negative_cpc_dict.items()})
                    
        else:
            # for the rows containing CPC, collect the (row name, column name) for cells that are negative
            negative_cpc_dict = {}
            for row_name, row in cpc_rows.iterrows():
                negative_cpc_assays = []
                for col_name, cell_value in row.items():
                    if 'negative' in str(cell_value).lower():
                        negative_cpc_assays.append(col_name)
                if negative_cpc_assays:
                    negative_cpc_dict[row_name] = negative_cpc_assays
            # create a df to store the results
            negative_cpc_df = pd.DataFrame({col: pd.Series(values) for col, values in negative_cpc_dict.items()})
            
        # return cpc_rvp_rows, cpc_p1_rows, cpc_p2_rows, negative_cpc_p1_assays, negative_cpc_p2_assays, p1_assays, p2_assays, rvp_assays, assays, cpc_rows, negative_cpc_df, negative_cpc_dict
        return negative_cpc_df
    
    def rnasep_check(self, binary_t13_df):
        # lowercase all column names and filter the col to find rnasep col
        binary_t13_df.columns = binary_t13_df.columns.str.lower()
        rnasep_cols_df = binary_t13_df.filter(like='rnasep', axis=1)

        # filter out the rows containing NTC, CPC, and NDC (as these are controls)
        # rnasep_cols_df = rnasep_cols_df[~rnasep_cols_df.index.str.contains('CPC|NTC|NDC', case=False)]

        # create a dictionary to store lists of sample names for each rnasep assay
        rnasep_assays_dict = {col: [] for col in rnasep_cols_df.columns}
        
        # go thru the samples (rows) in rnasep_cols_df
        for row_name, row in rnasep_cols_df.iterrows():
            for col_name, cell_value in row.items():
                if 'negative' in str(cell_value).lower():
                    # add the sample name in the dictionary corresponding to the col of rnasep_cols_df
                    rnasep_assays_dict[col_name].append(row_name)  
        # create a df to store the results
        neg_rnasep_df = pd.DataFrame({col: pd.Series(values) for col, values in rnasep_assays_dict.items()})

        return neg_rnasep_df 
    
    def ntc_check(self, assigned_sig_norm):
        # filter for columns t13, assay, and sample
        filtered_df = assigned_sig_norm[['t13', 'assay', 'sample']]
        # filter sample for anything that contains NTC (case-insensitive)
        ntc_filtered_df = filtered_df[filtered_df['sample'].str.contains('NTC', case=False, na=False)]
        # extract unique samples (NTCs) from ntc_filtered_df
        unique_samples = ntc_filtered_df['sample'].unique() # there shld not be duplicate samples, but good check
        # extract unique assays from ntc_filtered_df
        unique_assays = ntc_filtered_df['assay'].unique()
        # initialize dict with (sample, assay) as the key and t13 data as the value
        data_dict = {(sample, assay): None for sample in unique_samples for assay in unique_assays}
        # iterate thru ntc_filtered_df to collect t13 data IF it's > 0.5
        for assay in unique_assays:
            for sample in unique_samples:
                # filter the df for the current assay & filter the df for the current sample
                df = ntc_filtered_df[(ntc_filtered_df['sample'] == sample) & (ntc_filtered_df['assay'] == assay)]
                if not df.empty: # if sample_df is not empty
                    # get the t13 value
                    t13_val = df['t13'].values[0]  # accessing the first element
                    # check if t13 value is greater than 0.5 for any sample
                    if t13_val > 0.5: 
                        data_dict[(sample, assay)] = t13_val
                else:
                    print('Check if NTC samples were used in experiment and listed in assignment sheet.')
        # convert the data_dict into a dataframe
        high_raw_ntc_df = pd.Series(data_dict).reset_index()
        high_raw_ntc_df.columns = ['Sample', 'Assay', 't13']
        high_raw_ntc_df.dropna(subset=['t13'], inplace=True)
        high_raw_ntc_df.reset_index(inplace=True, drop=True)

        return high_raw_ntc_df
    
    def coinf_check(self, t13_hit_binary_output):
        
        # initialize dict to store coninf samples per assay
        coinf_samples_by_assay = {} # key is coinf sample, values are assays for which it is pos

        for _, row in t13_hit_binary_output.iterrows():
            
            # sum across the row to collect the number of total pos assays
            sumRow = row.sum() # input df is binary so pos for an assay is 1
            
            # if the sum is >2 (i.e. there is coinf btwn assays not incl RNaseP), then append all pos assays for that row to dict
            if sumRow > 2: # you expect sample to be pos for X assay and RNaseP

                # initialize a list to hold assays with value 1 for this sample
                positive_assays = []
        
                # iterate through each column (assay) to find assays with a value of 1
                for assay in t13_hit_binary_output.columns:
                    if row[assay] == 1:
                        positive_assays.append(assay)
        
                # Add the sample (row index) as the key and the list of assays as the value
                coinf_samples_by_assay[row.name] = positive_assays
       
        # convert coninf_samples_by_assay into a df for easy output
        coinf_df = pd.DataFrame.from_dict(coinf_samples_by_assay, orient='index')
        #coinf_df.drop('Summary')
        
        return coinf_df
    
    def no_crrna_check(self, binary_t13_df):
        # lowercase all column names and filter the col to find rnasep col
        binary_t13_df.columns = binary_t13_df.columns.str.lower()

        # Filter columns containing 'no_crrna'
        no_crrna_columns = binary_t13_df.columns[binary_t13_df.columns.str.contains("no_crrna")].tolist()
        
        # Create a dictionary to store the results
        no_crrna_samples = {}

        for col in no_crrna_columns:
            # Get the samples (row indices) with a value of 1 in this column
            samples = binary_t13_df[binary_t13_df[col] == 1].index.tolist()
            no_crrna_samples[col] = samples

        # Create a new DataFrame from the dictionary
        nocrrna_fail_df = pd.DataFrame(dict([(key, pd.Series(value)) for key, value in no_crrna_samples.items()]))

        return nocrrna_fail_df

    

            

          