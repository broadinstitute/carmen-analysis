import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import re

class RedCapper:
    def __init__(self):
        pass
    
    # method
    def build_redcap(self, fl_t13_hit_binary_output_2):

       
        ### convert 0 to 2 (negative)
        redcap_t13_hit_binary_output = fl_t13_hit_binary_output_2.replace(0, 2)

         ### drop any rows incl and below 'Summary' row
        if 'Summary' in redcap_t13_hit_binary_output.index:
            idx = redcap_t13_hit_binary_output.index.get_loc('Summary')
            redcap_t13_hit_binary_output = redcap_t13_hit_binary_output.iloc[:idx]

        ### convert any cell val with a dagger † to 6 (NTC contaminated)
        redcap_t13_hit_binary_output = redcap_t13_hit_binary_output.replace(r'.*†.*', 6, regex=True)

        ### convert col vals for invalid assays to 5 (invalid)
        # for all invalid samples
        redcap_t13_hit_binary_output.loc[redcap_t13_hit_binary_output['SAMPLE VALID? Y/N'] == 'N***', :] = 5

        # for all invalid assays
        assay_valid_cols = redcap_t13_hit_binary_output.columns[redcap_t13_hit_binary_output.loc['Assay Valid?'] == 'INVALID ASSAY']
        for col in assay_valid_cols:
            redcap_t13_hit_binary_output[col] = 5

        ### drop the 'SAMPLE VALID? Y/N' col
        redcap_t13_hit_binary_output = redcap_t13_hit_binary_output.drop('SAMPLE VALID? Y/N', axis=1)

        ### drop the 'Assay Valid?' row
        redcap_t13_hit_binary_output = redcap_t13_hit_binary_output.drop('Assay Valid?', axis=0)

        ### drop any columns containing no_crRNA
        redcap_t13_hit_binary_output = redcap_t13_hit_binary_output.loc[:, ~redcap_t13_hit_binary_output.columns.str.lower().str.contains('no_crrna')]

        ### strip all _ and asterisks from the column names
        for i, col in enumerate(redcap_t13_hit_binary_output.columns):
            if not re.search(r'rnasep|no_crrna', col, re.IGNORECASE):
                new_col = re.split(r'[_*]', col)[0]
                redcap_t13_hit_binary_output.columns.values[i] = new_col
            if  re.search(r'rnasep|no_crrna', col, re.IGNORECASE):
                new_col = re.split(r'[*]', col)[0]
                redcap_t13_hit_binary_output.columns.values[i] = new_col

        ### add columns for the assay that wasn't run with since REDCAP format needs all assays (RVP and BBP) headers in 
        bbp_assays = ['CCHFV', 'CHI', 'DENV', 'EBOV', 'HBV_DNA', 'HCV', 'HIV_1', 'HIV_2', 'HTV', 'LASV', 'MBV', 'MMV', 
                    'MPOX_DNA', 'ONN', 'PF_3_DNA', 'RBV', 'RVFV', 'SYPH_DNA', 'WNV', 'YFV', 'ZIKV']
        rvp_assays = ['SARS-COV-2', 'HCOV-HKU1', 'HCOV-NL63', 'HCOV-OC43', 'FLUAV', 'FLUBV', 'HMPV', 'HRSV', 'HPIV-3']
        # set column order
        column_order = bbp_assays + rvp_assays + ['RNASEP_P1','RNASEP_P2']
        # when adding the new columns, enter the value as 4 (not run)
        for col in column_order:
            if col not in redcap_t13_hit_binary_output.columns:
                redcap_t13_hit_binary_output[col] = 4
        
        # reorder cols
        redcap_t13_hit_binary_output = redcap_t13_hit_binary_output[column_order]
        

 




        





        return redcap_t13_hit_binary_output

