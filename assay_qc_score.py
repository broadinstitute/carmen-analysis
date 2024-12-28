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
        # filter the rows to find the CPCs in the df
        cpc_rows = t13_hit_binary_output[t13_hit_binary_output.index.str.contains('CPC')]

        # filter cpc_rows further and stratify into cpc_rvp, cpc_p1, cpc_p2
        cpc_rvp_rows = cpc_rows[cpc_rows.index.str.contains('_RVP', case=False)]
        cpc_p1_rows = cpc_rows[cpc_rows.index.str.contains('_P1', case=False)]
        cpc_p2_rows = cpc_rows[cpc_rows.index.str.contains('_P2', case=False)]

        for cpc, row in cpc_rvp_rows.iterrows():
            # extract the cpc_p1 suffix
            cpc_rvp_suffix = cpc.split('_')[-1].lower()
            # count the number of CPC_P1s -> this is tot_CPC_P1s (divisor)
            tot_CPC_RVPs = len(cpc_rvp_rows)
            # initialize a dict to hold CPC_P1 samples (value) with val 0 for assay (key)
            pos_CPC_RVPs = {}
            # count the number of CPC_P1s with val 0 -> neg_CPC_P1s
            for assay in cpc_rvp_rows.columns:
                assay_suffix = assay.split('_')[-1]
                if assay_suffix == cpc_rvp_suffix: # if sample is a CPC_P1 and assay is from P1
                    # initialize counter
                    counter=0
                    for _, row in cpc_rvp_rows.iterrows():
                        if row[assay] == 1: # if CPC_P1 is positive for that assay
                            counter+=1   
                    pos_CPC_RVPs[assay] = counter
                    # assign the pass/fail score to the assay
                    if pos_CPC_RVPs[assay]/tot_CPC_RVPs >= 0.5:
                        QC_score_per_assay_df.loc['QC3: CPC', assay] = 1 # pass
                    else:
                        QC_score_per_assay_df.loc['QC3: CPC', assay] = 0 # fail 

        for cpc, row in cpc_p1_rows.iterrows():
            # extract the cpc_p1 suffix
            cpc_p1_suffix = cpc.split('_')[-1].lower()
            # count the number of CPC_P1s -> this is tot_CPC_P1s (divisor)
            tot_CPC_P1s = len(cpc_p1_rows)
            # initialize a dict to hold CPC_P1 samples (value) with val 0 for assay (key)
            pos_CPC_P1s = {}
            # count the number of CPC_P1s with val 0 -> neg_CPC_P1s
            for assay in cpc_p1_rows.columns:
                assay_suffix = assay.split('_')[-1]
                if assay_suffix == cpc_p1_suffix: # if sample is a CPC_P1 and assay is from P1
                    # initialize counter
                    counter=0
                    for _, row in cpc_p1_rows.iterrows():
                        if row[assay] == 1: # if CPC_P1 is positive for that assay
                            counter+=1   
                    pos_CPC_P1s[assay] = counter
                    # assign the pass/fail score to the assay
                    if pos_CPC_P1s[assay]/tot_CPC_P1s >= 0.5:
                        QC_score_per_assay_df.loc['QC3: CPC', assay] = 1 # pass
                    else:
                        QC_score_per_assay_df.loc['QC3: CPC', assay] = 0 # fail 
        
        for cpc, row in cpc_p2_rows.iterrows():
            # extract the cpc_p1 suffix
            cpc_p2_suffix = cpc.split('_')[-1].lower()
            # count the number of CPC_P1s -> this is tot_CPC_P1s (divisor)
            tot_CPC_P2s = len(cpc_p2_rows)
            # initialize a dict to hold CPC_P1 samples (value) with val 0 for assay (key)
            pos_CPC_P2s = {}
            # count the number of CPC_P1s with val 0 -> neg_CPC_P1s
            for assay in cpc_p2_rows.columns:
                assay_suffix = assay.split('_')[-1]
                if assay_suffix == cpc_p2_suffix: # if sample is a CPC_P1 and assay is from P1
                    # initialize counter
                    counter=0
                    for _, row in cpc_p2_rows.iterrows():
                        if row[assay] == 1: # if CPC_P1 is positive for that assay
                            counter+=1   
                    pos_CPC_P2s[assay] = counter
                    # assign the pass/fail score to the assay
                    if pos_CPC_P2s[assay]/tot_CPC_P2s >= 0.5:
                        QC_score_per_assay_df.loc['QC3: CPC', assay] = 1 # pass
                    else:
                        QC_score_per_assay_df.loc['QC3: CPC', assay] = 0 # fail 

        """  
        ## OLD BELOW FOR GENERAL CPC TEST
        # filter the rows to find the CPCs in the df
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
        """      

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