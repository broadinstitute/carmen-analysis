import pandas as pd
import streamlit as st
class MedianSort:
     def __init__(self,crRNA_list ):
       if crRNA_list  == None: 
         self.crRNA_list = ('SARS-CoV-2', 'HCoV-HKU1', 'HCoV-NL63', 'HCoV-OC43', 
                  'FLUAV', 'FLUBV', 'HMPV', 'HRSV', 'HPIV-3', 'RnaseP', 'no-crRNA')
       else:
         self.crRNA_list = crRNA_list 
   
         

     def clean_matrix(self, df, sample_list):
         transposed_df = df.T
         # Rearrange the columns of the transposed DataFrame
         rearranged_df = transposed_df.reindex(sample_list, axis='columns')
         return rearranged_df

# Display the rearranged DataFrame

         
         return None
     
     def create_median(self, signal_norm):
         t_names = [col for col in signal_norm.columns if col.startswith('t')]
         #st.write(t_names)
         medians = signal_norm.groupby(['assay','sample'])
         
         med_frames = {}
         for name in t_names:
            time_med = medians[t_names].median()[name].unstack()
            time_med.index.names=['']
            time_med.columns.names=['']
            med_frames[name] = time_med
         final_med_frames = {}
         for timepoint in med_frames:
             #st.info(timepoint)
             final_med_frames[timepoint] = self.clean_matrix(med_frames[timepoint], self.crRNA_list)
             #st.dataframe(final_med_frames[timepoint] )
            
         return final_med_frames
         
