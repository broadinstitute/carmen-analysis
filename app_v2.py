#streamlit assets
import streamlit as st
from st_aggrid import AgGrid
from streamlit_option_menu import option_menu
#dataloading assetts
import pandas as pd
import numpy as np 
import math 
#plotting imports
import matplotlib.pyplot as plt 
import seaborn as sns 
#file imports
from io import BytesIO
import base64
from pathlib import Path
import os 
from os import path
import glob 
import re
from reader import DataReader
from norm import DataProcessor
from matcher import DataMatcher
from median_frame import MedianSort
from threshold import Thresholder

@st.cache_data
def convert_df(df):
    return df.to_csv(encoding='utf-8')

st.set_page_config(
    page_title="CARMEN",
    page_icon="👋",
)

st.sidebar.header("CARMEN Upload Page")


with st.sidebar:
    selected = option_menu("CARMEN RVP", ["data entry", 'outputs'], 
        icons=['play', 'play'], menu_icon="cast", default_index=0)
    

if selected == "data entry":
    path = Path(os.getcwd())
    path_dir = path
    all_files = set(map(os.path.basename, path_dir.glob('*')))
    assignment_files = sorted([fname for fname in all_files if (fname.endswith(".xlsx"))])
    data_files = sorted([fname for fname in all_files if (fname.endswith(".csv"))])
    # Instantiate the DataReader

    col1, col2 = st.columns(2)
    with col1: 
        ifc = st.selectbox('ChipDIM', ('96','192'), index = 1)
        st.session_state['ifc'] = ifc
    with col2: 
        instrument_type = st.selectbox("Instrument Type", ("BM", "EP1"), index =0)
        st.session_state['instrument_type'] =  instrument_type
    
    assignment_file = st.file_uploader("Upload Assignment File:", type=["xlsx"])
    if assignment_file is not None:
        file_like_object = BytesIO(assignment_file.read())
        assignment_df = pd.read_excel(file_like_object)
        #assignment_df = pd.read_excel(BytesIO(assignment_file.read()))
        st.session_state['assignment_file'] = assignment_file.name
        #st.dataframe(assignment_df)

        assignment_file_name = assignment_file.name

        match = re.match(r"^(\d+)_.*", assignment_file_name)
        barcode_assignment = ""
        if match:
            s= match.group(1)
        barcode_assignment = match.group(1)
        #st.warning(f"This is my data file name {assignment_file_name}")
        st.info(f"This is my IFC barcode: {barcode_assignment}")
    else:
        st.warning("Please upload an Assignment File.")
    
        
    # File uploader for the data file
    data_file = st.file_uploader("Upload Data File:", type=["csv"])
    if data_file is not None:
        file_like_object = BytesIO(data_file.read())
        df_data = pd.read_csv(file_like_object, on_bad_lines="skip")
        st.session_state['data_file'] = data_file.name
        #st.dataframe(df_data)

        reader = DataReader()

        phrases_to_find = [
            "Raw Data for Passive Reference ROX",
            "Raw Data for Probe FAM-MGB",
            "Bkgd Data for Passive Reference ROX",
            "Bkgd Data for Probe FAM-MGB"
        ]
        file_like_object.seek(0)
        # Extract dataframes from each CSV file
        data_file_name = data_file.name
        barcode = data_file_name.replace(".csv"," ")
        #st.warning(f"This is my data file name {data_file_name}")
        #st.warning(f"This is my IFC barcode: {barcode}")
        
        st.session_state['barcode'] = barcode
        dataframes = reader.extract_dataframes_from_csv(file_like_object, phrases_to_find)
        st.session_state['dataframes'] = dataframes
        for key, df in dataframes.items():            
                st.session_state[key] = df
                #st.success(f'Saved {key} dataframe Session State')
                #st.dataframe(df)
    else:
        st.warning("Please upload a Data File.")
    if data_file is not None and assignment_file is not None:
        processor = DataProcessor()
        normalized_dataframes = processor.background_processing(st.session_state['dataframes'])
        st.divider()
        st.success('Download Normalized Files Below:')
        for norm in normalized_dataframes:
            st.write(norm)
            df = normalized_dataframes[norm]
            csv = convert_df(df)
            st.download_button(
            f"Press to Download {norm}.csv",
            csv,
            f"{norm}.csv",
            "text/csv",
            key=f'download-{norm}'
            )
        st.divider()
        matcher = DataMatcher()
        assigned_norms, assigned_lists = matcher.assign_assays(assignment_file,normalized_dataframes['ref_norm'],normalized_dataframes['signal_norm'],st.session_state['ifc'])
        st.session_state['assigned_lists'] = assigned_lists
        st.divider()
        st.success('Download Normalized Assignment Files Below:')
        st.divider()
        for norm in assigned_norms:
            st.write(norm)
            df = assigned_norms[norm]
            csv = convert_df(df)
            st.download_button(
            f"Press to Download assigned_{norm}.csv",
            csv,
            f"assigned_{norm}.csv",
            "text/csv",
            key=f'download-assigned_{norm}'
            )

            median = MedianSort(crRNA_list=None)
            final_med_frames = median.create_median(assigned_norms['signal_norm_raw'])
            st.session_state['final_med_frames'] = final_med_frames 
        st.info(body = "click on outputs tab to visualize and get final data", icon="⬅️")
        
if selected == "outputs":
        timepoints = list(st.session_state['final_med_frames'].keys())

        # Slider for selecting the timepoint
        selected_timepoint = st.slider(
            label='Select Timepoint',
            min_value=1,
            max_value=len(timepoints),
            value=1,
            format="%d",
            step=1,
            help="Slide to select different timepoints."
        )

            # Get the selected timepoint

        timepoint = timepoints[selected_timepoint-1]

            # Display the dataframe
        st.write(f"Displaying data for timepoint: {timepoint}")
        #print(st.session_state['final_med_frames'][timepoint].head())
        st.dataframe(st.session_state['final_med_frames'][timepoint], height=(len(st.session_state['final_med_frames'][timepoint]) + 1) * 35 + 3, use_container_width=True)
        last_key = list(st.session_state['final_med_frames'].keys())[-1]
        csv = convert_df(st.session_state['final_med_frames'][last_key])
        #print(st.session_state['final_med_frames'][timepoint].head())

        st.download_button(
            f"Press to Download t13.csv",
            csv,
            f"t13.csv",
            "text/csv",
            key=f'download-t13'
            )
        
        tresholder  = Thresholder()
        tresholder.raw_thresholder(st.session_state['assigned_lists']['assay_list'],st.session_state['final_med_frames'][last_key]) 
        

