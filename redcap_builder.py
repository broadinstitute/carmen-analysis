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
    def build_redcap(self, fl_t13_hit_binary_output_2, date, barcode_assignment):

        # merge rows with the same sampleid_prefix - keep assay results unique, combine record_ids, update sampleid
        def merge_group(group):
            # select first row in subset of redcap_t13_hit_binary_output grouped by sampleid_prefix
            merged_row = pd.DataFrame(columns=group.columns)
            merged_row.loc[0] = group.iloc[0]

            # the group is the unique sampleid_prefix - each group should have max 2 rows
            for col in group.columns:
                if col not in ["record_id", "date", "ifc", "sampleid", "sampleid_prefix"]:
                    # if merged_row['cchfv'] = [5,4], then lambda fn will produce [5,None]
                    # dropna will make it merged_row['cchfv'] = [5]
                    # .unique ensures that only unique vals are retained
                   
                    if all(group[col] == 4):
                        merged_row[col] = 4
                    else: # if group[col] = [5,4] or [3, 4] - there's no world where it would be [5,3]
                        filtered_values = group.loc[group[col] != 4, col].dropna().unique()
                        merged_row[col] = filtered_values[0] if len(filtered_values) ==1 else filtered_values[1]
                     
            # each record_id is split and the unique panel suffixes are added to suffix_record_id 
            merged_row['suffix_record_id'] = '_'.join(group['record_id'].apply(lambda x: x.split('_')[-1]).unique())

            # assign a sampleid to the merged_row (doesn't matter as sampleid col will be dropped later)
            #merged_row['sampleid'] = group['sampleid'].iloc[0]

            return merged_row
       
        ### format input flagged t13 binary hit file 
        redcap_t13_hit_binary_output = fl_t13_hit_binary_output_2.copy()
        redcap_t13_hit_binary_output = redcap_t13_hit_binary_output.astype(str)

        ### convert 0 to 2 (negative)
        redcap_t13_hit_binary_output = redcap_t13_hit_binary_output.replace(to_replace=r'^0.*', value=2, regex=True)

        ### drop any rows incl and below 'Summary' row
        if 'Summary' in redcap_t13_hit_binary_output.index:
            idx = redcap_t13_hit_binary_output.index.get_loc('Summary')
            redcap_t13_hit_binary_output = redcap_t13_hit_binary_output.iloc[:idx]

        ### convert any cell val with a dagger † to 6 (NTC contaminated)
        """ 
        test_redcap_t13_hit_binary_output = redcap_t13_hit_binary_output.copy() 
        test_redcap_t13_hit_binary_output = test_redcap_t13_hit_binary_output.astype(str)
        test_redcap_t13_hit_binary_output = test_redcap_t13_hit_binary_output.map(lambda x: str(x) if not isinstance(x, str) else x)
         
        for _, row in high_raw_ntc_signal_df.iterrows():
            cont_ntc_sample = row['Sample']
            cont_ntc_assay = row['Assay'].upper()
            
            # Check if the sample exists in the row index and the assay exists in the column header
            if cont_ntc_sample in test_redcap_t13_hit_binary_output.index and cont_ntc_assay in test_redcap_t13_hit_binary_output.columns:
                current_value = test_redcap_t13_hit_binary_output.loc[cont_ntc_sample, cont_ntc_assay]
                if '†' in current_value:
                    test_redcap_t13_hit_binary_output.loc[cont_ntc_sample, cont_ntc_assay] = '6'
        """
        redcap_t13_hit_binary_output = redcap_t13_hit_binary_output.astype(str)
        redcap_t13_hit_binary_output = redcap_t13_hit_binary_output.map(lambda x: '6' if '†' in x else x)
        

        
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
        rvp_assays = ['SARS_COV-2', 'HCOV_HKU1', 'HCOV_NL63', 'HCOV_OC43', 'FLUAV', 'FLUBV', 'HMPV', 'HRSV', 'HPIV_3']
        # set column order
        column_order = bbp_assays + rvp_assays + ['RNASEP_P1','RNASEP_P2']
        # when adding the new columns, enter the value as 4 (not run)
        for col in column_order:
            if col not in redcap_t13_hit_binary_output.columns:
                redcap_t13_hit_binary_output[col] = 4
       
        ### reorder cols
        redcap_t13_hit_binary_output = redcap_t13_hit_binary_output[column_order]

        ### add in the metadata columns
        # date
        redcap_t13_hit_binary_output.insert(0, "date", date)
        # barcode assignment
        redcap_t13_hit_binary_output.insert(1, "ifc", barcode_assignment)
        # sampleid
        sampleid = []
        for idx in redcap_t13_hit_binary_output.index: # strip all _ and asterisks from the sample names
            cleaned_idx = re.sub(r'[\*\|†\s]', '', idx)
            sampleid.append(cleaned_idx)
        redcap_t13_hit_binary_output.insert(2, "sampleid", sampleid)
        # recordid
        record_id = []
        for row in redcap_t13_hit_binary_output.itertuples():
            samp_id = row.sampleid 
            record_id_val = barcode_assignment + '_' + samp_id 
            record_id.append(record_id_val)

        redcap_t13_hit_binary_output.insert(0, "record_id", record_id)

        ### merge same samples ran on different panels 
        # extract sampleid before panel _P1 or _P2 or _RVP
        redcap_t13_hit_binary_output['sampleid_prefix'] = redcap_t13_hit_binary_output['sampleid'].str.replace(r'(_P1|_P2|_RVP)$', '', regex=True)
       
        # subset redcap into two dfs 
        controlsDF = redcap_t13_hit_binary_output[redcap_t13_hit_binary_output['sampleid'].str.contains('NDC|CPC|NTC', regex=True, na=False)]
        samplesDF = redcap_t13_hit_binary_output[~redcap_t13_hit_binary_output['sampleid'].str.contains('NDC|CPC|NTC', regex=True, na=False)]

        # apply the merge_group function to each group in the groupby obj (which is a df)
        samplesDF = samplesDF.groupby('sampleid_prefix').apply(merge_group).reset_index(drop=True)

        # fix the suffix in record_id
        record_id_fix = []
        for row in samplesDF.itertuples():
            record_id = row.record_id 
            suffix_record_id = row.suffix_record_id
            record_id = record_id.split("_")[:-1] 
            record_id = "_".join(record_id) 
            new_record_id = record_id + "_" + suffix_record_id
            record_id_fix.append(new_record_id)
        samplesDF['record_id'] = record_id_fix  

        # drop suffix_record_id
        samplesDF = samplesDF.drop(columns=['suffix_record_id'])

        # concatenate back to redcap
        concat_redcap_t13_hit_binary_output = pd.concat((samplesDF, controlsDF), axis=0, ignore_index=True)

        ### write sampleid as the sample_prefix for all samples but those containing CPC, NTC, and NDC
        mask = ~concat_redcap_t13_hit_binary_output['sampleid'].str.contains('NTC|CPC|NDC', regex=True, na=False)
        concat_redcap_t13_hit_binary_output.loc[mask, 'sampleid'] = concat_redcap_t13_hit_binary_output['sampleid_prefix']

        # drop sample_prefix_id
        concat_redcap_t13_hit_binary_output = concat_redcap_t13_hit_binary_output.drop(columns=['sampleid_prefix'])

        ### lowercase all columns in redcap_t13_hit_binary_output for REDCAP data entry
        concat_redcap_t13_hit_binary_output.columns = concat_redcap_t13_hit_binary_output.columns.str.lower()

        ### reset index
        concat_redcap_t13_hit_binary_output = concat_redcap_t13_hit_binary_output.reset_index(drop=True)


        return concat_redcap_t13_hit_binary_output # redcap_t13_hit_binary_output, samplesDF, controlsDF

