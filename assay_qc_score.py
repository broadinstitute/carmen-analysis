import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class Assay_QC_Score:
    def __init__(self):
        pass

    def assay_level_score(self, t13_hit_binary_output):
     ## prepare a df to contain the QC scores per assay tested    
         # lowercase all column names and filter the col to find rnasep col
        t13_hit_binary_output.columns = t13_hit_binary_output.columns.str.lower()
        # initialize df to hold QC scores (QC1-4 as rows, assays as cols)
        QC_score_per_assay_df = pd.DataFrame(index=['QC1: NTC', 'QC2: NDC', 'QC3: CPC', 'QC4: RNaseP'],
                                             columns=t13_hit_binary_output.columns)
     
     ## for assay score #1 NTC:
        # filter the rows to find the NDCs in the df
        ntc_rows = t13_hit_binary_output[t13_hit_binary_output.index.str.contains('NTC')]
        # count the number of NDCs -> this is tot_NDCs (divisor)
        tot_NTCs = len(ntc_rows)
        # initialize a dict to hold NDC samples (value) with val 0 for assay (key)
        neg_NTCs = {}
        # count the number of NDCs with val 0 -> neg_NDCs
        for assay in ntc_rows.columns:
            # initialize counter
            counter=0
            for _, row in ntc_rows.iterrows():
                if row[assay] == 0:
                    counter+=1   
            neg_NTCs[assay] = counter

            if neg_NTCs[assay]/tot_NTCs == 1.0:
                QC_score_per_assay_df.loc['QC1: NTC', assay] = 1 # complete pass
            
            if 0.75 < neg_NTCs[assay]/tot_NTCs < 1.0:
                QC_score_per_assay_df.loc['QC1: NTC', assay] = 0.75 # high-tier pass

            if 0.5 < neg_NTCs[assay]/tot_NTCs <= 0.75:
                QC_score_per_assay_df.loc['QC1: NTC', assay] = 0.5 # mid-tier pass

            if 0.25 < neg_NTCs[assay]/tot_NTCs <= 0.5:
                QC_score_per_assay_df.loc['QC1: NTC', assay] = 0.25 # low-tier pass
            
            if neg_NTCs[assay]/tot_NTCs <= 0.25:
                QC_score_per_assay_df.loc['QC1: NTC', assay] = 0 # fail
        
     ## for assay score #2 NDC:
        # filter the rows to find the NDCs in the df
        ndc_rows = t13_hit_binary_output[t13_hit_binary_output.index.str.contains('NDC')]
        # count the number of NDCs -> this is tot_NDCs (divisor)
        tot_NDCs = len(ndc_rows)
        # initialize a dict to hold NDC samples (value) with val 0 for assay (key)
        neg_NDCs = {}
        # count the number of NDCs with val 0 -> neg_NDCs
        for assay in ndc_rows.columns:
            # initialize counter
            counter=0
            for _, row in ndc_rows.iterrows():
                if row[assay] == 0:
                    counter+=1   
            neg_NDCs[assay] = counter
        
            if neg_NDCs[assay]/tot_NDCs > 0.5:
                QC_score_per_assay_df.loc['QC2: NDC', assay] = 1 # pass
            else:
                QC_score_per_assay_df.loc['QC2: NDC', assay] = 0 # fail
            
     ## for assay score #3 CPC:
        # filter the rows to find the NDCs in the df
        cpc_rows = t13_hit_binary_output[t13_hit_binary_output.index.str.contains('CPC')]
        # count the number of NDCs -> this is tot_NDCs (divisor)
        tot_CPCs = len(cpc_rows)
        # initialize a dict to hold NDC samples (value) with val 0 for assay (key)
        pos_CPCs = {}
        # count the number of NDCs with val 0 -> neg_NDCs
        for assay in cpc_rows.columns:
            # initialize counter
            counter=0
            for _, row in cpc_rows.iterrows():
                if row[assay] == 1:
                    counter+=1   
            pos_CPCs[assay] = counter
        
            if pos_CPCs[assay]/tot_CPCs >= 0.5:
                QC_score_per_assay_df.loc['QC3: CPC', assay] = 1 # pass
            else:
                QC_score_per_assay_df.loc['QC3: CPC', assay] = 0 # fail       

     ## for assay score #$ RNaseP:
        # filter the cols to find the RNaseP col in the df
        rnasep_df = t13_hit_binary_output.filter(like='rnasep', axis=1)
        # count the number of rows -> this is tot_samples (divisor)
        tot_samples = len(rnasep_df)
        # initialize a dict to hold samples (value) with val 1 for RNaseP (key)
        pos_RNasePs = {}
        # for rounding:
        decimals = 4
        # count the number of samples with val 1 for RNaseP -> pos_RNasePs
        for col in rnasep_df.columns: #shld be just 1 col
            # initialize counter
            counter=0
            for _, sample in rnasep_df.iterrows():
                if sample[col] == 1:
                    counter+=1
            pos_RNasePs[col] = counter
            # score is 0.XX, so if all samples are pos for rnasep - QC4 score is 1.0
            QC_score_per_assay_df.loc['QC4: RNaseP', col] = round(((pos_RNasePs[col])/tot_samples), decimals)

     ## sum down columns to produce "Final Score" per assay    
        # sum the cols of QC_score_per_assay_df - this generates a series
        final_score_series = QC_score_per_assay_df.sum(axis=0, skipna=True)
        # create final_score_df
        final_score_df = pd.DataFrame(final_score_series).transpose()
        final_score_df.index = ['Final Score']

        # concatenate the final_score_df to QC_score_per_assay_df
        QC_score_per_assay_df= pd.concat([QC_score_per_assay_df,final_score_df], axis=0)

        for col in QC_score_per_assay_df.columns:
            QC_score_per_assay_df.loc['Final Score', col] = round(QC_score_per_assay_df.loc['Final Score', col], decimals)
       
    
     ## return csv containing df with QC no. and score
        return QC_score_per_assay_df