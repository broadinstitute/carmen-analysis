import numpy as np 
import pandas as pd 

class Thresholder:
    def __init__(self):
        pass

    def convert_df(self,df):
        return df.to_csv(index=False).encode('utf-8')
    
    def raw_thresholder(self, t13_df):

        # Filter rows containing 'NTC' in any column, produces df
        ntc_PerAssay = t13_df[t13_df.index.str.contains('NTC')]

        # Collapse any columns that have the same name in the NTCs df 
        ntc_PerAssay = ntc_PerAssay.T.groupby(by=ntc_PerAssay.columns).mean()

        # Calculate mean of each column
        mean_ntc = ntc_PerAssay.mean(axis=1)

        # Make a new df with these mean values
        ntc_mean_df = pd.DataFrame({'NTC Mean': mean_ntc})

        # Transpose the results to have assays as columns
        ntc_mean_df = ntc_mean_df.transpose()

        # Scale the NTC mean by 1.8 to generate thresholds
        raw_thresholds_df = ntc_mean_df * 1.8

        # Append the threshold to ntc_mean_df
        ntc_mean_df = pd.concat([ntc_mean_df, raw_thresholds_df], ignore_index=True, axis=0)
        ntc_mean_df.index = ['NTC Mean', 'NTC Threshold']


        # Binary
        binary = ['positive', 'negative']
       
        for col_name, col_data in t13_df.items():
            for index, value in col_data.items():
                threshold = raw_thresholds_df.loc['NTC Mean', col_name]

                t13_df[col_name] = t13_df[col_name].map(str)
                # Compare the value with the threshold and assign positive/negative accordingly
                if value >= threshold:
                    t13_df.at[index, col_name] = binary[0]
                else:
                    t13_df.at[index, col_name] = binary[1]
       
        # Create a new row called 'Summary' at the bottom of the hit output sheet
        t13_df.loc['Summary'] = t13_df.apply(lambda col: col.value_counts().get('positive', 0))
 
        return ntc_mean_df, t13_df

    