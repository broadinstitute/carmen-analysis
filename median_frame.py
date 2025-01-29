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

        # initialize a dictionary
        med_frames = {}
        # for each timepoint
        for name in t_names: 
            # groups signal_norm by assay and sample
            # in groupby, pandas automatically sorts the assay col in alpha order
            # selects the column for the specific timepoint
            # calculate the median value for each unique group (assay and sample) for that specific timepoint
            # finally .unstack() unstacks the sample (outer layer of grouping), so the samples are columns and assays are rows
            time_med = signal_norm.groupby(['assay', 'sample'])[name].median().unstack()
            # clean up the names of index and columns
            time_med.index.names=['']
            time_med.columns.names=['']
            # transpose time_med so the rows are samples and the columns are assays
            time_med_transposed = time_med.transpose()
            # store transposed df for each timepoint into dictionary med_frames with timepoint as the key
            med_frames[name] = time_med_transposed

        return med_frames
      
          
     




