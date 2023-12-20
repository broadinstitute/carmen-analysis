import numpy as np
import pandas as pd

class Summarized: 
    def __init__(self):
        pass

    def summarizer(self, binary_t13_df):
        # Create a new row called 'Summary'
        binary_t13_df.loc['Summary'] = binary_t13_df.apply(lambda col: col.value_counts().get('positive', 0))
 
        positive_samples_dict ={}

        for col_name, col_data in binary_t13_df.items():
           pos_samples_list = col_data.index[col_data == 'positive'].tolist()
           positive_samples_dict[col_name] = pos_samples_list

        positive_samples_df = pd.DataFrame()
        for key, value in positive_samples_dict.items():
            # Create a pandas Series for each key, flatten the list
            series = pd.Series(value, name=key)
            # Concatenate the series to the DataFrame
            positive_samples_df = pd.concat([positive_samples_df, series], axis=1)

        positive_samples_df.loc['Totals'] = binary_t13_df.loc['Summary']

        totals_row = positive_samples_df.loc[['Totals']]
        positive_samples_df = positive_samples_df.drop('Totals')
        positive_samples_df = pd.concat([totals_row, positive_samples_df])
        
        return positive_samples_df



        


  