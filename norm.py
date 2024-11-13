import pandas as pd
class DataProcessor:
    def __init__(self):
        pass  # No initialization parameters needed as of now
    def background_processing(self, dataframes):
        # Substract the background from the probe and reference data
        probe_norm = dataframes["probe_raw"].subtract(dataframes["probe_bkgd"]) #from FAM
        ref_norm = dataframes["ref_raw"].subtract(dataframes["ref_bkgd"]) #from Rox
        
         
        # Normalize the probe signal with the reference dye signal
        signal_norm = pd.DataFrame(probe_norm /ref_norm )
        signal_norm = signal_norm.reset_index()
        
        # split Column ID into SampleID and AssayID for signal
        splitassignment_signal = signal_norm['Chamber ID'].str.split("-",n=1,expand=True)
        signal_norm["sampleID"] = splitassignment_signal[0]
        signal_norm["assayID"] = splitassignment_signal[1]
        signal_norm = signal_norm.set_index('Chamber ID')
        
        # split Column ID into SampleID and AssayID for ref
        ref_norm = ref_norm.reset_index()
        splitassignment_ref = ref_norm['Chamber ID'].str.split("-",n=1,expand=True)
        ref_norm["sampleID"] = splitassignment_ref[0]
        ref_norm["assayID"] = splitassignment_ref[1]
        ref_norm = ref_norm.set_index('Chamber ID')
         
        norm_outputs = {}
        norm_outputs['signal_norm'] = signal_norm
        norm_outputs['ref_norm'] = ref_norm

        
        return norm_outputs
