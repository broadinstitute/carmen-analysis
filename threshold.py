import numpy as np 
import pandas as pd 
import streamlit as st
class Thresholder:
    def __init__(self):
        pass

    def convert_df(self,df):
        return df.to_csv(index=False).encode('utf-8')
    
    def raw_thresholder(self, assay_list,t13_df):
        #st.dataframe(assay_list)
        #st.data_editor(t13_df)

        # Filter rows containing 'NTC' in any column
        ntc_rows = t13_df[t13_df.index.str.contains('NTC')]
        #st.dataframe(ntc_rows)
        

        # Compute the mean of the filtered rows
        raw_thresholds = ntc_rows.mean() * 1.8
        
        '''
        #st.dataframe(t13_df)
        csv = self.convert_df(assay_list)
        st.download_button(
        f"Press to Download t13.csv",
        csv,
        f"t13.csv",
        "text/csv",
        key=f'download-t13'
        )
        '''
        return assay_list