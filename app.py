import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from st_aggrid import AgGrid
from pathlib import Path
import os 
from os import path
import glob 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

tgap = 1 
time_assign = {}
for cycle in range(1,38):
    tpoint = "t" + str(cycle)
    time_assign[tpoint] = tgap + 3 + (cycle-1) * 5
# calculate the real timing of image
# used for image and axis labeling

def gettime(tname):
    tname = "t" + str(tname)
    realt = time_assign[tname]
    return (realt)


def load_data(ifc, instrument_type, csv_file ):
    if instrument_type == 'BM':
        if ifc == '96':
            probe_df = pd.read_csv(csv_file,header = 18449, nrows = 9216) # FAM
            reference_df = pd.read_csv(csv_file, header = 9231, nrows = 9216) # ROX
            bkgd_ref_df = pd.read_csv(csv_file, header = 27667, nrows = 9216)
            bkgd_probe_df = pd.read_csv(csv_file,header = 36885, nrows = 9216)
        if ifc == '192':
            probe_df = pd.read_csv(csv_file,header = 9233, nrows = 4608)
            reference_df = pd.read_csv(csv_file, header = 4623, nrows = 4608)
            bkgd_ref_df = pd.read_csv(csv_file, header = 13843, nrows = 4608)
            bkgd_probe_df = pd.read_csv(csv_file,header = 18453, nrows = 4608)
    elif instrument_type == 'EP1':
        if ifc == '192':
            probe_df = pd.read_csv(csv_file,header = 9238, nrows = 4608)
            reference_df = pd.read_csv(csv_file, header = 4628, nrows = 4608)
            bkgd_ref_df = pd.read_csv(csv_file, header = 18458, nrows = 4608)
            bkgd_probe_df = pd.read_csv(csv_file,header = 23068, nrows = 4608)
    return probe_df, reference_df, bkgd_ref_df, bkgd_probe_df

def clean_data(probe_df, reference_df, bkgd_ref_df, bkgd_probe_df, count_tp):
    c_to_drop = 'Unnamed: ' + str(count_tp+1)
    probe_df = probe_df.set_index("Chamber ID").drop(c_to_drop,axis=1)
    reference_df = reference_df.set_index("Chamber ID").drop(c_to_drop,axis=1)
    bkgd_ref_df = bkgd_ref_df.set_index("Chamber ID").drop(c_to_drop,axis=1)
    bkgd_probe_df = bkgd_probe_df.set_index("Chamber ID").drop(c_to_drop,axis=1)

    probe_df.columns = probe_df.columns.str.lstrip() # remove spaces from beginning of column names
    reference_df.columns = reference_df.columns.str.lstrip()
    bkgd_ref_df.columns = bkgd_ref_df.columns.str.lstrip()
    bkgd_probe_df.columns = bkgd_probe_df.columns.str.lstrip()

    # rename column names
    probe_df.columns = ['t' + str(col) for col in probe_df.columns]
    reference_df.columns = ['t' + str(col) for col in reference_df.columns]
    bkgd_ref_df.columns = ['t' + str(col) for col in bkgd_ref_df.columns]
    bkgd_probe_df.columns = ['t' + str(col) for col in bkgd_probe_df.columns]

    return probe_df, reference_df, bkgd_ref_df, bkgd_probe_df

def get_signal(probe_df, reference_df, bkgd_ref_df, bkgd_probe_df, layout_file, count_tp, out_folder, exp_name, instrument_type):
    # Substract the background from the probe and reference data
    probe_bkgd_substracted = probe_df.subtract(bkgd_probe_df)
    ref_bkgd_substracted = reference_df.subtract(bkgd_ref_df)

    # Normalize the probe signal with the reference dye signal
    signal_df = pd.DataFrame(probe_bkgd_substracted/ref_bkgd_substracted)

    # reset index
    signal_df = signal_df.reset_index()
    # split Column ID into SampleID and AssayID
    splitassignment = signal_df['Chamber ID'].str.split("-",n=1,expand=True)
    signal_df["sampleID"] = splitassignment[0]
    signal_df["assayID"] = splitassignment[1]

    #set index again to Chamber ID
    signal_df = signal_df.set_index('Chamber ID')

    sampleID_list = signal_df.sampleID.unique()
    assayID_list = signal_df.assayID.unique()

    # Save csv
    signal_df.to_csv(path.join(out_folder, exp_name+ '_' +instrument_type + '_1_signal_bkgdsubtracted_norm_' + str(count_tp) +'.csv'))

    # Create two dictionaries that align the IFC wells to the sample and assay names
    samples_layout = pd.read_excel(path.join('',layout_file),sheet_name='layout_samples').applymap(str)
    assays_layout = pd.read_excel(path.join('',layout_file),sheet_name='layout_assays').applymap(str)

    # Create a dictionary with assay/sample numbers and their actual crRNA / target name
    assays = pd.read_excel(path.join('',layout_file),sheet_name='assays')
    assay_dict = dict(zip(assays_layout.values.reshape(-1),assays.values.reshape(-1)))

    samples = pd.read_excel(path.join('',layout_file),sheet_name='samples')
    samples_dict = dict(zip(samples_layout.values.reshape(-1),samples.values.reshape(-1)))

    # Map assay and sample names
    signal_df['assay'] = signal_df['assayID'].map(assay_dict)
    signal_df['sample'] = signal_df['sampleID'].map(samples_dict)

    # Save csv
    signal_df.to_csv(path.join(out_folder, exp_name+'_' +instrument_type +'_2_signal_bkgdsubtracted_norm_named_' + str(count_tp) +'.csv'))
    return assays, assay_dict, samples, samples_dict, signal_df

def create_tlist(ifc, count_tp, signal_df, assays, samples):
    t_names = []
    for i in range(1,count_tp+1):
        t_names.append(('t' + str(i)))

    # Create a list of all assays and samples
    # only indicate columns with unique assays. np.unique could be used on the list, but messes up our prefered the order
    if ifc == '96':
        new_array = np.stack(assays[['C1','C2','C3','C4','C5']].values,axis=-1)
    if ifc == '192':
        new_array = np.stack(assays[['C1','C2', 'C3']].values,axis=-1)

    assay_list = np.concatenate(new_array).tolist()
    st.write('identified crRNAs: ',len(assay_list))

    # Do this for the samples
    if ifc == '96':
        chambers = ['C{}'.format(i) for i in range(1, 13)]
        new_array = np.stack(samples[chambers].values,axis=-1)
    if ifc == '192':
        chambers = ['C{}'.format(i+1) for i in range(24)]
        new_array = np.stack(samples[chambers].values,axis=-1)
    #     new_array = np.stack(samples[['C1','C2','C3','C4','C5','C6',\
    #                                   'C7','C8','C9','C10','C11','C12']].values,axis=-1)

    sample_list = np.concatenate(new_array).tolist()
    st.write('identified samples: ',len(sample_list))

    # Grouped medians
    st.dataframe(signal_df)
    #st.write(signal_df.columns)
    grouped_df = signal_df.groupby(['assay','sample'])
    tlist = []
    for t in range(count_tp):
        tlist.append("t" + str(t+1))
    st.dataframe(grouped_df[tlist].median())
    #medians = signal_df.groupby(['assay','sample']).median()
    med_frames = {}
    for name in t_names:
        time_med = grouped_df[tlist].median()[name].unstack()
        time_med.index.names=['']
        time_med.columns.names=['']
        med_frames[name] = time_med
    return med_frames, assay_list, sample_list

def plt_heatmap(df_dict, samplelist, assaylist, tp,exp_name, out_folder, instrument_type):
    frame = df_dict[f't{tp}'][samplelist].reindex(assaylist)
    fig, axes = plt.subplots(1,1,figsize=(len(frame.columns.values)*0.5,len(frame.index.values)*1.0))
    ax = sns.heatmap(frame,cmap='Reds',square = True,cbar_kws={'pad':0.002}, annot_kws={"size": 20})
    rt = gettime(tp)
    plt.title(exp_name +' '+str(rt)+'min - median values', size = 28)
    plt.xlabel('Samples', size = 14)
    plt.ylabel('Assays', size = 14)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.tick_params(axis="y", labelsize=16)
    ax.tick_params(axis="x", labelsize=16)
    plt.yticks(rotation=0)

    tgt_num = len(samplelist)
    gd_num = len(assaylist)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    h_lines = np.arange(3,gd_num,3)
    v_lines = np.arange(3,tgt_num,3)
    axes.hlines(h_lines, colors = 'silver',alpha=0.9,linewidths = 0.35,*axes.get_xlim())
    axes.vlines(v_lines, colors = 'silver',alpha=0.9,linewidths = 0.35,*axes.get_ylim())
    st.pyplot(fig)
    plt.savefig(exp_name + '_'+instrument_type +'2_heatmap_'+str(tp)+'.png', format='png', bbox_inches='tight', dpi=400)

st.set_page_config(
    page_title="CARMEN RVP",
    page_icon="👋",
)

st.markdown("# CARMEN RVP")
st.sidebar.header("CARMEN RVP")


with st.sidebar:
    selected = option_menu("CARMEN RVP", ["data entry", 'analysis'], 
        icons=['play', 'play'], menu_icon="cast", default_index=0)
    

if selected == "data entry":
    path = Path(os.getcwd())
    path_dir = path
    all_files = set(map(os.path.basename, path_dir.glob('*')))
    assignment_files = sorted([fname for fname in all_files if (fname.endswith(".xlsx"))])
    data_files = sorted([fname for fname in all_files if (fname.endswith(".csv"))])

    col1, col2 = st.columns(2)
    with col1: 
        ifc = st.selectbox('ChipDIM', ('96','192'), index = 1)
        st.session_state['ifc'] = ifc
    with col2: 
        instrument_type = st.selectbox("Instrument Type", ("BM", "EP1"), index =0)
        st.session_state['instrument_type'] =  instrument_type
    
    assignment_file = st.selectbox(label="Assignment File:", options=assignment_files)
    st.session_state['assignment_file'] =  assignment_file
    assignment_df = pd.read_excel(assignment_file)
    st.dataframe(assignment_df)
    
    data_file = st.selectbox(label="Data File:", options=data_files)
    df_data = pd.read_csv(data_file, on_bad_lines="skip")
    st.session_state['data_file'] =  data_file
    st.dataframe(df_data)
elif selected == 'analysis':
    probe_df, reference_df, bkgd_ref_df, bkgd_probe_df = load_data(ifc=st.session_state["ifc"], instrument_type = st.session_state['instrument_type'],csv_file=st.session_state["data_file"] )
    timepoints = len(probe_df.columns.tolist()[1:])-1
    st.session_state['timepoints'] =  timepoints
    st.success('Loading Data', icon="✅")
    probe_df, reference_df, bkgd_ref_df, bkgd_probe_df = clean_data(probe_df, reference_df, bkgd_ref_df, bkgd_probe_df, count_tp = timepoints)
    st.success('Cleaning Data', icon="✅")
    assays, assay_dict, samples, samples_dict, signal_df = get_signal(probe_df, reference_df, bkgd_ref_df, bkgd_probe_df, layout_file = st.session_state['assignment_file'], count_tp = timepoints, out_folder = os.getcwd(), exp_name = "test", instrument_type=st.session_state['instrument_type'])
    st.success('Signal Data Produced', icon="✅")
    med_frames, assay_list, sample_list = create_tlist(ifc=st.session_state["ifc"], count_tp=timepoints, signal_df=signal_df, assays=assays, samples=samples)
    st.success('Get timelist', icon="✅")
    st.write(timepoints)
    plt_heatmap(df_dict=med_frames, samplelist= sample_list, assaylist=assay_list, tp=timepoints, exp_name="test", out_folder="carmen_test", instrument_type= st.session_state['instrument_type'])
    st.success('Plotting', icon="✅")
    