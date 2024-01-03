import pandas as pd

class MedianSort:
     def __init__(self,crRNA_list ):
       if crRNA_list is None: 
         self.crRNA_list = ('SC2', 'HCoV-HKU1', 'HCoV-NL63', 'HCoV-OC43', 
                  'FLUAV', 'FLUBV', 'HMPV', 'HRSV', 'HPIV-3', 'RnaseP', 'no-crRNA')
       else:
         self.crRNA_list = crRNA_list 
   

# Display the rearranged DataFrame
     
     def create_median(self, signal_norm):
         # create a list containing t1 thru t13
        t_names = [col for col in signal_norm.columns if col.startswith('t')]

        med_frames = {}
        for name in t_names:
            time_med = signal_norm.groupby(['assay', 'sample'])[name].median().unstack()
            time_med.index.names=['']
            time_med.columns.names=['']
            time_med_transposed = time_med.transpose()
            med_frames[name] = time_med_transposed

        return med_frames
      
          
     




