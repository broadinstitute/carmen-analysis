import pandas as pd
class DataProcessor:
    def __init__(self):
        pass  # No initialization parameters needed as of now
    def background_processing(self, dataframes):
        # Substract the background from the probe and reference data
        probe_norm = dataframes["probe_raw"].subtract(dataframes["probe_bkgd"])
        ref_norm = dataframes["ref_raw"].subtract(dataframes["ref_bkgd"])
        
        # Normalize the probe signal with the reference dye signal
        signal_norm = pd.DataFrame(probe_norm /ref_norm )
        signal_norm = signal_norm.reset_index()
        
        # split Column ID into SampleID and AssayID for signal
        splitassignment_signal = signal_norm['Chamber ID'].str.split("-",n=1,expand=True)
        signal_norm["sampleID"] = splitassignment_signal[0]
        signal_norm["assayID"] = splitassignment_signal[1]
        signal_norm = signal_norm.set_index('Chamber ID')
        
        # split Column ID into SampleID and AssayID for norm
        ref_norm = ref_norm.reset_index()
        splitassignment_ref = ref_norm['Chamber ID'].str.split("-",n=1,expand=True)
        ref_norm["sampleID"] = splitassignment_ref[0]
        ref_norm["assayID"] = splitassignment_ref[1]
        ref_norm = ref_norm.set_index('Chamber ID')
        
        norm_outputs = {}
        norm_outputs['signal_norm'] = signal_norm
        norm_outputs['ref_norm'] = ref_norm
       
        '''
        #save
        signal_df.to_csv(out_folder+exp_name+ '_' +instrument_type + '_1_signal_bkgdsubtracted_norm_' + str(count_tp) +'.csv')

        # Do the  same for reference df
        reference_df = reference_df.reset_index()
        splitassignment = reference_df['Chamber ID'].str.split("-",n=1,expand=True)
        reference_df["sampleID"] = splitassignment[0]
        reference_df["assayID"] = splitassignment[1]
        reference_df = reference_df.set_index('Chamber ID')
        #save
        reference_df.to_csv(out_folder+exp_name+ '_' +instrument_type + '_1_reference_' + str(count_tp) +'.csv')

        print(max(ref_bkgd_substracted))
        '''
        return norm_outputs
