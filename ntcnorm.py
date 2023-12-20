import numpy as np
import pandas as pd

class Normalized:
    def __init__(self):
        pass

    def normalizr(self, t13_df):
        
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

        # Divide all values in t13_df by the NTC mean for that assay
        for col_name, col_data in t13_df.items():
            for index, value in col_data.items():
                # Find the NTC mean for the assay
                ntc_mean = ntc_mean_df.loc['NTC Mean', col_name]
                # Divide the value by the NTC mean per assay
                t13_df.at[index, col_name] = value/ntc_mean
        
        return t13_df
    
    

        


