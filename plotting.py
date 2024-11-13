import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class Plotter:
    def __init__(self):
        pass
    
    def gettime(time_assign, timepoints):
        realt = time_assign[timepoints]
        return (realt)

    def plt_heatmap(self, tgap, barcode_number, df_dict, sample_list, assay_list, tp):
        # Create a dictonary for timepoints
        time_assign = {}

        for cycle in range(1,len(tp)+1):
            tpoint = "t" + str(cycle)
            time_assign[tpoint] = tgap + 3 + (cycle-1) * 5
        
        fig_timepoints = {}
        half_samples = int(len(sample_list)/2)

        if len(sample_list) == 192: 
            # Split the sample list into two halves
            first_half_samples = sample_list[:half_samples]
            second_half_samples = sample_list[half_samples:]

            for i in tqdm(tp):
                df_dict[i] = df_dict[i].transpose()
                # Split heatmap into two subplots (2-row, 1-column layout)
                fig, axes = plt.subplots(2, 1, figsize=(len(first_half_samples) * 0.5, len(assay_list) * 0.5 * 2))

                # First heatmap (first 96 samples)
                frame1 = df_dict[i][first_half_samples].reindex(assay_list)
                ax1 = sns.heatmap(frame1, cmap='Reds', square=True, cbar_kws={'pad': 0.002}, annot_kws={"size": 20}, ax=axes[0])
                ax1.set_title(f'Heatmap for {barcode_number} at {time_assign[i]} minutes (First {half_samples} Samples)', size=28)
                ax1.set_xlabel('Samples', size=14)
                ax1.set_ylabel('Assays', size=14)
                bottom, top = ax1.get_ylim()
                ax1.set_ylim(bottom + 0.5, top - 0.5)
                ax1.tick_params(axis="y", labelsize=16)
                ax1.tick_params(axis="x", labelsize=16)
                plt.yticks(rotation=0)

                # Second heatmap (next 96 samples)
                frame2 = df_dict[i][second_half_samples].reindex(assay_list)
                ax2 = sns.heatmap(frame2, cmap='Reds', square=True, cbar_kws={'pad': 0.002}, annot_kws={"size": 20}, ax=axes[1])
                ax2.set_title(f'Heatmap for {barcode_number} at {time_assign[i]} minutes (Next {half_samples} Samples)', size=28)
                ax2.set_xlabel('Samples', size=14)
                ax2.set_ylabel('Assays', size=14)
                bottom, top = ax2.get_ylim()
                ax2.set_ylim(bottom + 0.5, top - 0.5)
                ax2.tick_params(axis="y", labelsize=16)
                ax2.tick_params(axis="x", labelsize=16)
                plt.yticks(rotation=0)

                 # Adjust layout
                plt.tight_layout()

                # Save the figure to the dictionary
                fig_timepoints[i] = fig

        else: 
            for i in tqdm(tp):
                df_dict[i] = df_dict[i].transpose()
                frame = df_dict[i][sample_list].reindex(assay_list)
                fig, axes = plt.subplots(1,1,figsize=(len(frame.columns.values)*0.5,len(frame.index.values)*0.5))
                ax = sns.heatmap(frame,cmap='Reds',square = True,cbar_kws={'pad':0.002}, annot_kws={"size": 20})

                # calculate the real timing of the image
                rt = time_assign[i]

                plt.title('Heatmap for 'f'{barcode_number} at '+str(rt)+' minutes', size = 28)
                plt.xlabel('Samples', size = 14)
                plt.ylabel('Assays', size = 14)
                bottom, top = ax.get_ylim()
                ax.set_ylim(bottom + 0.5, top - 0.5)
                ax.tick_params(axis="y", labelsize=14)
                ax.tick_params(axis="x", labelsize=14)
                plt.yticks(rotation=0)

                tgt_num = len(sample_list)
                gd_num = len(assay_list)
                bottom, top = ax.get_ylim()
                ax.set_ylim(bottom + 0.5, top - 0.5)
                h_lines = np.arange(3,gd_num,3)
                v_lines = np.arange(3,tgt_num,3)
                axes.hlines(h_lines, colors = 'silver',alpha=0.9,linewidths = 0.35,*axes.get_xlim())
                axes.vlines(v_lines, colors = 'silver',alpha=0.9,linewidths = 0.35,*axes.get_ylim())
                fig_timepoints[i] = fig

        return fig_timepoints
    
    def t13_plt_heatmap(self, tgap, barcode_number, df, sample_list, assay_list, tp):
        # Create a dictonary for timepoints
        time_assign = {}

        for cycle in range(1,len(tp)+1):
            tpoint = "t" + str(cycle)
            time_assign[tpoint] = tgap + 3 + (cycle-1) * 5
        last_key = list(time_assign.keys())[-1] 

        half_samples = int(len(sample_list)/2)

        if len(sample_list) == 192: 
            # Split the sample list into two halves
            first_half_samples = sample_list[:half_samples]
            second_half_samples = sample_list[half_samples:]

            fig, axes = plt.subplots(2, 1, figsize=(len(first_half_samples) * 0.5, len(assay_list) * 0.5 * 2))

            df = df.transpose()

            # First heatmap (first 96 samples)
            frame1 = df[first_half_samples].reindex(assay_list)
            ax1 = sns.heatmap(frame1, cmap='Reds', square=True, cbar_kws={'pad': 0.002}, annot_kws={"size": 20}, ax=axes[0])
            ax1.set_title(f'Heatmap for data normalized against assay-specific NTC mean, for {barcode_number} at {time_assign[last_key]} minutes (First {half_samples} Samples)', size=22)
            ax1.set_xlabel('Samples', size=16)
            ax1.set_ylabel('Assays', size=16)
            bottom, top = ax1.get_ylim()
            ax1.set_ylim(bottom + 0.5, top - 0.5)
            ax1.tick_params(axis="y", labelsize=14)
            ax1.tick_params(axis="x", labelsize=12)
            plt.yticks(rotation=0)

            # Second heatmap (next 96 samples)
            frame2 = df[second_half_samples].reindex(assay_list)
            ax2 = sns.heatmap(frame2, cmap='Reds', square=True, cbar_kws={'pad': 0.002}, annot_kws={"size": 20}, ax=axes[1])
            ax2.set_title(f'Heatmap for data normalized against assay-specific NTC mean, for {barcode_number} at {time_assign[last_key]} minutes (Next {half_samples} Samples)', size=28)
            ax2.set_xlabel('Samples', size=16)
            ax2.set_ylabel('Assays', size=16)
            bottom, top = ax2.get_ylim()
            ax2.set_ylim(bottom + 0.5, top - 0.5)
            ax2.tick_params(axis="y", labelsize=14)
            ax2.tick_params(axis="x", labelsize=12)
            plt.yticks(rotation=0)

            # Adjust layout
            plt.tight_layout()

        else: 
            df = df.transpose()
            frame = df[sample_list].reindex(assay_list)
            fig, axes = plt.subplots(1,1,figsize=(len(frame.columns.values)*0.5,len(frame.index.values)*0.5))
            ax = sns.heatmap(frame,cmap='Reds',square = True,cbar_kws={'pad':0.002}, annot_kws={"size": 20})

            plt.title(f'Heatmap for data normalized against assay-specific NTC mean, for {barcode_number} at {time_assign[-1]} minutes', size = 28)
            plt.xlabel('Samples', size = 14)
            plt.ylabel('Assays', size = 14)
            bottom, top = ax.get_ylim()
            ax.set_ylim(bottom + 0.5, top - 0.5)
            ax.tick_params(axis="y", labelsize=16)
            ax.tick_params(axis="x", labelsize=16)
            plt.yticks(rotation=0)

            tgt_num = len(sample_list)
            gd_num = len(assay_list)
            bottom, top = ax.get_ylim()
            ax.set_ylim(bottom + 0.5, top - 0.5)
            h_lines = np.arange(3,gd_num,3)
            v_lines = np.arange(3,tgt_num,3)
            axes.hlines(h_lines, colors = 'silver',alpha=0.9,linewidths = 0.35,*axes.get_xlim())
            axes.vlines(v_lines, colors = 'silver',alpha=0.9,linewidths = 0.35,*axes.get_ylim())
    
        return fig