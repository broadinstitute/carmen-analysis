import numpy as np 
import pandas as pd 
import sys

class Thresholder:
    def __init__(self):
        pass

    def convert_df(self,df):
        return df.to_csv(index=False).encode('utf-8')
    
    def raw_thresholder(self, unique_crRNA_assays, assigned_only, t13_df, CLI_thresh_arg):

        # Filter rows containing 'NTC' in any column, produces df
        ntc_PerAssay = t13_df[t13_df.index.str.contains('NTC')]

        # Collapse any columns that have the same name in the NTCs df 
        # there should not be duplicates for NTCs in t13_df - as this would have been taken care of by MedianSort
        # but the .mean() below is to ensure this - but more so simplify the groupby and transpose functions
        ntc_PerAssay = ntc_PerAssay.T.groupby(by=ntc_PerAssay.columns).mean()

        # Calculate mean of each column
        mean_ntc = ntc_PerAssay.mean(axis=1)

        # Make a new df with these mean values
        ntc_mean_df = pd.DataFrame({'NTC Mean': mean_ntc})

        # Transpose the results to have assays as columns
        ntc_mean_df = ntc_mean_df.transpose()

        if CLI_thresh_arg == '1.8_Mean':
            
            # Scale the NTC mean by 1.8 to generate thresholds
            raw_thresholds_df = ntc_mean_df * 1.8

            # Calculate the Normalized NTC threshold too
            norm_thresholds_df = raw_thresholds_df / ntc_mean_df

            # Append ntc_mean_df, raw_thresholds_df, and norm_thresholds_df
            raw_thresholds_df = pd.concat([ntc_mean_df, raw_thresholds_df, norm_thresholds_df], ignore_index=True, axis=0) 
            raw_thresholds_df.index = ['NTC Mean', 'NTC Threshold', 'Normalized NTC Threshold']

        elif CLI_thresh_arg == '3_SD':
            
            # Calculate the standard deviation per assay of all NTC values (replicates included)
            filtered_assigned_only = assigned_only.loc[:,["t13","assay","sample"]]
            filtered_assigned_only = filtered_assigned_only[filtered_assigned_only['sample'].str.contains('NTC')]

            ntc_sd_df = pd.DataFrame(index = ['SD'], columns = unique_crRNA_assays)
        
            for i in unique_crRNA_assays:
                filtered_assigned_only_i = filtered_assigned_only[filtered_assigned_only['assay'].isin([i])]
                t13_data_i = filtered_assigned_only_i['t13']
                for col_name in ntc_sd_df.columns:
                    if i == col_name: 
                        ntc_sd_df.at['SD',col_name] = t13_data_i.std()

            # raw_thresholds_df = ntc_sd_df
                        
            # Make a copy of ntc_mean_df
            raw_thresholds_df = ntc_mean_df.copy(deep=True)

            # Scale the SD by 3 to generate 3*SD per assay
            ntc_3sd_df = ntc_sd_df * 3
            ntc_3sd_df.index = ['NTC 3*SD']

            # Sum the NTC mean and 3*SD per assay to generate thresholds
            for col_name, col_data in raw_thresholds_df.items():
                for index, value in col_data.items():
                    # Find the NTC 3*SD for the given assay
                    ntc_3sd = ntc_3sd_df.loc['NTC 3*SD', col_name]
                    # Sum the mean and the 3*SD value
                    raw_thresholds_df.at[index, col_name] = value + ntc_3sd

            # Append the threshold to ntc_mean_df
            raw_thresholds_df = pd.concat([ntc_mean_df, ntc_sd_df, ntc_3sd_df, raw_thresholds_df], ignore_index=True, axis=0)
            raw_thresholds_df.index = ['NTC Mean', 'NTC Standard Deviation', 'NTC 3*SD', 'NTC Threshold']

            # If there is only 1 NTC for an assay, we cannot take the std dev and these rows remain blank in raw_thresholds_df
            # So we resort to using the 1.8_Mean for these instances.
            for col in raw_thresholds_df.columns:
                if pd.isna(raw_thresholds_df.at['NTC Threshold', col]):
                    ntc_mean = raw_thresholds_df.at['NTC Mean', col]
                    raw_thresholds_df.at['NTC Threshold', col] = 1.8 * ntc_mean

            # Calculate the Normalized NTC threshold too
            norm_thresholds_df = raw_thresholds_df.loc['NTC Threshold'] / ntc_mean_df
            raw_thresholds_df = pd.concat([raw_thresholds_df, norm_thresholds_df], ignore_index=True, axis=0)
            raw_thresholds_df.index = ['NTC Mean', 'NTC Standard Deviation', 'NTC 3*SD', 'NTC Threshold', 'Normalized NTC Threshold']
     
        else:
            print("Consult ReadME and input appropriate command-line arguments to specify thresholding method.")
            sys.exit()


        # Produce the t13 output as Binary Pos/Neg Results
        binary = ['POSITIVE', 'NEGATIVE']
       
        for col_name, col_data in t13_df.items():
            for index, value in col_data.items():
                threshold = raw_thresholds_df.loc['NTC Threshold', col_name]

                t13_df[col_name] = t13_df[col_name].map(str)
                # Compare the value with the threshold and assign positive/negative accordingly
                if value >= threshold:
                   t13_df.at[index, col_name] = binary[0]
                else:
                    t13_df.at[index, col_name] = binary[1]
       
        # Create a new row called 'Summary' at the bottom of the hit output sheet
        t13_df.loc['Summary'] = t13_df.apply(lambda col: col.value_counts().get('POSITIVE', 0))
 
        return raw_thresholds_df, t13_df

    