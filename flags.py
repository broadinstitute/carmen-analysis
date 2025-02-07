import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import re

class Flagger:
    def __init__(self):
        pass
    
    # method
    def assign_flags(self, fail_nocrRNA_check_df, high_raw_ntc_signal_df, rnasep_df, QC_score_per_assay_df, t13_hit, t13_quant_norm, pos_samples_df, ntc_thresh, t13_hit_binary):
    
        files = [t13_hit, t13_quant_norm, pos_samples_df, ntc_thresh, t13_hit_binary] # 0, 1, 2, 3, 4
        flagged_files = [] # store modified files after applying flags

        for i, file in enumerate(files): 

            ### CPC flags 
            ## need to be added to t13_hit_output, rounded_t13_quant_norm, summary_samples_df, ntc_thresholds_output, t13_hit_binary_output
            flagged_file = file.copy() # work on a copy of the orig file
            invalid_assays = []  #  track which assays are invalid based on QC3 test results
            for row in QC_score_per_assay_df.itertuples():
                if row.Index == 'QC3: CPC':
                    for assay in QC_score_per_assay_df.columns: 
                        score = row._asdict().get(assay, None)
                        #score = getattr(row, assay)
                        if score == 0 and 'no_crrna' not in assay.lower(): # CPC test has failed, the assay is invalid
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
                        invalid_row.append('VALID ASSAY') # this way, invalid_row has same dimensions as flagged_file's cols

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
            else: 
                # create the "Assay Valid?" row with default value 'VALID ASSAY'
                valid_assay_row = ['VALID ASSAY'] * len(flagged_file.columns)
                valid_assay_row_df = pd.DataFrame([valid_assay_row], columns=flagged_file.columns)
                valid_assay_row_df.index = ["Assay Valid?"]
                # concatenate valid_assay_row_df to file_df
                data = flagged_file.iloc[0:]  
                flagged_file = pd.concat([valid_assay_row_df, data], ignore_index=False) # concatenate all
            


            if i == 2: # summary_samples_df
                for col in flagged_file.columns:
                    if 'INVALID ASSAY' in flagged_file[col].values:  # check if any cell in the column contains 'INVALID ASSAY'
                        valid_values = ["This assay is considered invalid due to failing Quality Control Test #3, which evaluates performance of the Combined Positive Control sample."]
                        flagged_file[col] = flagged_file[col].apply(lambda x: '' if x != 'INVALID ASSAY' and x not in valid_values else x)
                        
        
            ### NTC flags
            ## dagger flag needs to be added to t13_hit_output, rounded_t13_quant_norm, summary_samples_df, t13_hit_binary_output
            if i in {0,1,2,4}:
                processed_samples = set()

                for _, row in high_raw_ntc_signal_df.iterrows(): # cols are Sample, Assay, t13 
                        cont_ntc_sample = row['Sample'].upper() # NEG NTC sample
                        cont_ntc_sample = cont_ntc_sample.strip()
                        cont_ntc_assay = row['Assay'].upper() # NTC assay
                        # now iterate over the flagged file
                        for idx, sample_row in flagged_file.iterrows(): 
                            idx = str(idx).strip().upper()
                            if cont_ntc_sample == idx:
                                # add † to each cell value
                                for assay_col in flagged_file.columns:  
                                    if assay_col.strip().upper() == cont_ntc_assay:
                                        # check that the sample-assay pair has alr been processed
                                        if (cont_ntc_sample, cont_ntc_assay) not in processed_samples:
                                            processed_samples.add((cont_ntc_sample, cont_ntc_assay))
                                            # check if the value is NA (NaN)
                                            if pd.isna(sample_row[assay_col]) or sample_row[assay_col] == '':
                                                flagged_file.loc[idx, assay_col] = '†'  # only dagger if value is NA
                                            else:
                                                flagged_file[assay_col] = flagged_file[assay_col].astype(str)
                                                #flagged_file.at[idx, assay_col] = str(flagged_file.at[idx, assay_col])
                                                flagged_file.at[idx, assay_col] = f"{sample_row[assay_col]}†"  # add dagger to the value
                """  
                for _, row in high_raw_ntc_signal_df.iterrows():
                    for col in high_raw_ntc_signal_df.columns: # cols are Sample, Assay, t13 
                        cont_ntc_sample = row['Sample'].upper() # NEG NTC sample
                        cont_ntc_sample = cont_ntc_sample.strip()
                        cont_ntc_assay = row['Assay'].upper() # NTC assay
                        # now iterate over the flagged file
                        for idx, sample_row in flagged_file.iterrows(): 
                            if cont_ntc_sample == str(idx).upper():
                                # add † to each cell value
                                for assay_col in flagged_file.columns:  
                                    if assay_col.upper() == cont_ntc_assay.upper():
                                        # check that the sample-assay pair has alr been processed
                                        if (cont_ntc_sample, cont_ntc_assay) not in processed_samples:
                                            processed_samples.add((cont_ntc_sample, cont_ntc_assay))
                                            # check if the value is NA (NaN)
                                            if pd.isna(sample_row[assay_col]) or sample_row[assay_col] == '':
                                                flagged_file.loc[idx, assay_col] = '†'  # only dagger if value is NA
                                            else:
                                                flagged_file[assay_col] = flagged_file[assay_col].astype(str)
                                                #flagged_file.at[idx, assay_col] = str(flagged_file.at[idx, assay_col])
                                                flagged_file.at[idx, assay_col] = f"{sample_row[assay_col]}†"  # add dagger to the value
                """    
                for _, row in high_raw_ntc_signal_df.iterrows(): 
                    for col in high_raw_ntc_signal_df.columns: 
                        cont_ntc_sample = row['Sample'] # NEG NTC sample
                        cont_ntc_assay = row['Assay'] # NTC assay
                        # now iterate over the flagged file
                        for idx, sample_row in flagged_file.iterrows(): 
                            if cont_ntc_sample == idx: 
                                # add † to sample name
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
                
            ## dagger and explanation needs to be added to ntc_thresholds_output
            if i == 3:
                for col in flagged_file.columns:
                    if flagged_file.at['NTC Mean', col] == 0.5: 
                        # add a dagger to the col name (assay)
                        new_col_name = f'{col}†'
                        flagged_file.rename(columns={col: new_col_name}, inplace=True)
                        # add a dagger to the value in the cell
                        flagged_file.at["NTC Mean", new_col_name] = f'{flagged_file.at["NTC Mean", new_col_name]}†'    

                # add legend at the bottom of the file
                legend_added = False
                for col in flagged_file.columns:
                    if '†' in str(col):
                        label = ("All NTC samples for this assay were excluded from threshold calculations due to potential contamination. " 
                                 "A threshold of 0.5 a.u. was assigned as this is the maximum tolerable signal for an NTC control.")
                        cont_NTC_thresh_legend_label = pd.DataFrame(data=[[label] + [pd.NA]*(len(flagged_file.columns) - 1)], columns=flagged_file.columns, index=["Legend for †:"])
                        cont_NTC_thresh_legend_label_filled = cont_NTC_thresh_legend_label.fillna('')
                        # concatenate the invalid_legend label df to file df 
                        flagged_file = pd.concat([flagged_file, cont_NTC_thresh_legend_label_filled], ignore_index=False) # concatenate
                        legend_added = True
                        break                    
                        
            ### no-crRNA flags
            ## need to be added to t13_hit, t13_quant_norm, pos_samples_df, t13_hit_binary
            if i in {0,1,4}: # t13_hit, t13_quant_norm,t13_hit_binary
               
                summary_rows = flagged_file.index[flagged_file.index.str.contains('Summary', na=False)].tolist()
                
                summary_index = flagged_file.index.get_loc(summary_rows[0])
               
                df1 = flagged_file.iloc[:summary_index].copy()  # Everything before 'Summary'
                df2 = flagged_file.iloc[summary_index:].copy()  # Everything from 'Summary' onward (includes 'Summary')

                ## treat df1
                # make the Sample Valid column and the default val is Y
                df1.insert(0, 'Sample Valid? Y/N','')
                string_index = df1.index.to_series().astype(str)

                # assign 'Y' only to samples
                df1.loc[~string_index.str.contains(r'Assay Valid?'), 'Sample Valid? Y/N'] = 'Y'
              
                # assing 'N' to samples that have failed the no_crRNA check
                fail_nocrRNA_check_df.columns = fail_nocrRNA_check_df.columns.str.upper() # upper all cols before iterating
                
                for _, row in fail_nocrRNA_check_df.iterrows():
                    for no_crrna_assay in fail_nocrRNA_check_df.columns: 
                        sample = row[no_crrna_assay] # no_crrna_assay won't have an * denoting its invalid
                        for index,_ in df1.iterrows():
                            for col in df1.columns:
                                # find the sample in flagged_file
                                if sample == index:
                                    # add N and *** to the Sample Valid? col
                                    df1.at[index,'Sample Valid? Y/N'] = 'N***'

                                    # add *** to the value in the cell of sample, no_crRNA assay
                                    stripped_columns = df1.columns.str.rstrip('*')
                                    if no_crrna_assay in stripped_columns:
                                        original_column = df1.columns[stripped_columns.get_loc(no_crrna_assay)]
                                        if not str(df1.at[index, original_column]).endswith('***'):
                                            df1.at[index, original_column] = f'{df1.at[index, original_column]}***'

                                    #if no_crrna_assay in df1.columns.str.rstrip('*'):
                                    #    if not str(df1.at[index, col]).endswith('***'):
                                    #        df1.at[index, no_crrna_assay] = f'{df1.at[index, no_crrna_assay]}***'
                
                ## treat df2 and make df3
                df2_cols = df2.columns
                df3_cols = ['Sample Valid? Y/N'] + list(df2_cols)
                df3 = pd.DataFrame(index=['Summary'], columns=df3_cols)
                # populate df3 using df2's Summary row
                if 'Summary' in df2.index:
                    summary_row = df2.loc['Summary']
                    for df2_col in df2_cols:
                        df3.at['Summary', df2_col] = summary_row[df2_col]
                # make sure this value is blank 
                df3.at['Summary', 'Sample Valid? Y/N'] = ''

                # filter df2 to exclude rows containing 'Summary' and add a column at the end
                df2_allElse = df2[~df2.index.str.contains('Summary', na=False)].copy()
                #df2_allElse.loc[:, ""] = " " 
                df2_allElse.insert(len(df2_allElse.columns), "", " ") # add an unnamed empty column at the very end
                df2_allElse.columns = ['Sample Valid? Y/N'] + list(df2_allElse.columns[:-1]) # rename the existing columns

                # add df3 and df2_allElse together
                df3 = pd.concat((df3, df2_allElse), axis=0, ignore_index=False)

                ## concatenate df1 and df2 to reform the flagged_file
                flagged_file = pd.concat([df1, df3], axis=0, ignore_index= False)

                # add legend at the bottom of the file
                legend_added = False
                for index, sample_row in flagged_file.iterrows():
                    if '***' in sample_row['Sample Valid? Y/N']:
                        label = 'This sample is invalid due to testing positive against the no-crRNA assay, an included negative assay control.'
                        fail_nocrrna_legend_label = pd.DataFrame(data=[[label] + [pd.NA]*(len(flagged_file.columns) - 1)], columns=flagged_file.columns, index=["Legend for ***:"])
                        fail_nocrrna_legend_label_filled = fail_nocrrna_legend_label.fillna('')
                        # concatenate the invalid_legend label df to file df 
                        flagged_file = pd.concat([flagged_file, fail_nocrrna_legend_label_filled], ignore_index=False) # concatenate
                        legend_added = True
                        break
             
            if i == 2: 
                invalid_samples = pd.concat([fail_nocrRNA_check_df[col] for col in fail_nocrRNA_check_df.columns]).dropna().unique()

                # iterate over the flagged_file df and mark the matching samples with '***'
                for sample in invalid_samples:
                    flagged_file = flagged_file.apply(lambda col: col.map(lambda x: f'{sample}***' if x == sample else x))
                
                # add legend at the bottom of the file
                legend_added = False
                for index, sample_row in flagged_file.iterrows():
                    if sample_row.apply(lambda x: '***' in str(x)).any():
                        label = 'This sample is invalid due to testing positive against the no-crRNA assay, an included negative assay control.'
                        fail_nocrrna_legend_label = pd.DataFrame(data=[[label] + [pd.NA]*(len(flagged_file.columns) - 1)], columns=flagged_file.columns, index=["Legend for ***:"])
                        fail_nocrrna_legend_label_filled = fail_nocrrna_legend_label.fillna('')
                        # concatenate the invalid_legend label df to file df 
                        flagged_file = pd.concat([flagged_file, fail_nocrrna_legend_label_filled], ignore_index=False) # concatenate
                        legend_added = True
                        break
          
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

        return invalid_assays, invalid_samples, flagged_files, processed_samples, cont_ntc_sample, cont_ntc_assay
