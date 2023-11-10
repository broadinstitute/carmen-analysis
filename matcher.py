import pandas as pd 
import numpy as np
import streamlit as st
from os import path
import io

class DataMatcher:
    def __init__(self):
        pass  # No initialization parameters needed as of now
# Create two dictionaries that align the IFC wells to the sample and assay names

    def extract_and_stack(self, df):
    # Extract columns that start with 'C'
        c_columns = [col for col in df.columns if col.startswith('C')]
        # Stack the values and convert to a flat list
        stacked_array = np.stack(df[c_columns].values, axis=-1)
        flat_list = np.concatenate(stacked_array).tolist()
        return flat_list
    


    def assign_assays(self,assignment_file,ref_norm,signal_norm,ifc):
        #content_io = io.StringIO(assignment_file)
        samples_layout = pd.read_excel(assignment_file,sheet_name='layout_samples').applymap(str)
        assays_layout = pd.read_excel(assignment_file,sheet_name='layout_assays').applymap(str)

        # Create a dictionary with assay/sample numbers and their actual crRNA / target name
        assays = pd.read_excel(assignment_file,sheet_name='assays')
        assay_dict = dict(zip(assays_layout.values.reshape(-1),assays.values.reshape(-1)))

        samples = pd.read_excel(assignment_file,sheet_name='samples')
        samples_dict = dict(zip(samples_layout.values.reshape(-1),samples.values.reshape(-1)))

        #st.write(samples_dict)
        #st.write(assay_dict)

        # mapp assay and sample names to signal df
        signal_norm['assay'] = signal_norm['assayID'].map(assay_dict)
        signal_norm['sample'] = signal_norm['sampleID'].map(samples_dict)

        # mapp assay and sample names to reference df
        ref_norm['assay'] = ref_norm['assayID'].map(assay_dict)
        ref_norm['sample'] = ref_norm['sampleID'].map(samples_dict)


        assigned_norms = {}
        assigned_norms['signal_norm_raw'] = signal_norm
        assigned_norms['ref_norm_raw'] = ref_norm
        
        
        assays = pd.read_excel(assignment_file,sheet_name='assays')
        samples = pd.read_excel(assignment_file,sheet_name='samples')

        # Create a list of all assays and samples

        #matching layout assays to assays 
        assay_list = self.extract_and_stack(assays)
        #mathching layout samples to samples 
        samples_list = self.extract_and_stack(samples)
        assigned_lists = {}
        assigned_lists['assay_list'] = assay_list 
        assigned_lists['samples_list'] = samples_list 
        st.info(f"Number of crRNAs: {len(assay_list)}")
        st.info(f"Number of identified samples: {len(samples_list)}")
        return assigned_norms, assigned_lists
