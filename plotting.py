import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import matplotlib.patches as patches

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
                # Add space between the two subplots (vertical spacing)
                plt.subplots_adjust(hspace=1)

                # First heatmap (first 96 samples)
                frame1 = df_dict[i][first_half_samples].reindex(assay_list)
                annot1 = frame1.map(lambda x: 'X' if (pd.isna(x) or x == 'NaN' or x is None) else '')
                ax1 = sns.heatmap(frame1, cmap='Reds', square=True, cbar_kws={'pad': 0.002}, annot = None, fmt='', annot_kws={"size": 1000, "color": "black"}, ax=axes[0], 
                                  linewidths = 1, linecolor = "black")
                
                # Track x-axis labels that need a dagger
                dagger_labels = set()
                
                # Add cross-hatches for "X"-marked cells
                if not annot1.empty: 
                    for y in range(annot1.shape[0]):
                        for x in range(annot1.shape[1]):
                            if annot1.iloc[y, x] == 'X':
                                # Calculate cell coordinates
                                x_start, y_start = x, y
                                x_end, y_end = x + 1, y + 1

                                # Add cross-hatches
                                ax1.add_line(plt.Line2D([x_start, x_end], [y_start, y_end], color='black', linewidth=1.5))  # Top-left to bottom-right
                                ax1.add_line(plt.Line2D([x_start, x_end], [y_end, y_start], color='black', linewidth=1.5))  # Bottom-left to top-right

                                # Collect x-axis labels that correspond to the "X"
                                dagger_labels.add(frame1.columns[x])
                    
                    # Modify x-axis labels to include daggers
                    x_labels = ax1.get_xticklabels()
                    new_labels = [
                        f"† {label.get_text()}" if label.get_text() in dagger_labels else label.get_text()
                        for label in x_labels
                    ]
                    ax1.set_xticklabels(new_labels, rotation=90, ha='right')
                    # Place the legend below the first heatmap
                    left1, right1 = ax1.get_xlim()
                    top1, bottom1 = ax1.get_ylim() 
                    ax1.text(left1, top1 + 7, 
                                '†: The NTC sample for this assay was removed from the analysis due to potential contamination.', 
                                ha='left', fontsize=12, style='italic')
                               
                # Adjust layout
                ax1.set_title(f'Heatmap for {barcode_number} at {time_assign[i]} minutes (Plate #1: {half_samples} Samples)', size=28)
                ax1.set_xlabel('Samples', size=18)
                ax1.set_ylabel('Assays', size=18)
                top1, bottom1 = ax1.get_ylim() 
                ax1.set_ylim(top1 + 0.25, bottom1 - 0.25)
                left1, right1 = ax1.get_xlim()
                ax1.set_xlim(left1 - 0.25, right1 + 0.25)
                ax1.tick_params(axis="y", labelsize=16, width = 2, length = 5)
                ax1.tick_params(axis="x", labelsize=16, width = 2, length = 5)
                plt.yticks(rotation=0)
                plt.tight_layout()
                ax1.axhline(y=top1 + 0.16, color='k',linewidth=6)
                ax1.axhline(y=bottom1 - 0.14, color='k',linewidth=6)
                ax1.axvline(x=left1 - 0.14, color='k',linewidth=6)
                ax1.axvline(x=right1 + 0.15, color='k',linewidth=6)
                
                # Second heatmap (next 96 samples)
                frame2 = df_dict[i][second_half_samples].reindex(assay_list)
                annot2 = frame2.map(lambda x: 'X' if (pd.isna(x) or x == 'NaN' or x is None) else '')
                ax2 = sns.heatmap(frame2, cmap='Reds', square=True, cbar_kws={'pad': 0.002}, annot = None, annot_kws={"size": 20}, ax=axes[1], 
                                  linewidths = 1, linecolor = "black")
                
                # Track x-axis labels that need a dagger
                dagger_labels = set()
                
                # Add cross-hatches for "X"-marked cells
                if not annot2.empty: 
                    for y in range(annot2.shape[0]):
                        for x in range(annot2.shape[1]):
                            if annot2.iloc[y, x] == 'X':
                                # Calculate cell coordinates
                                x_start, y_start = x, y
                                x_end, y_end = x + 1, y + 1

                                # Add cross-hatches
                                ax2.add_line(plt.Line2D([x_start, x_end], [y_start, y_end], color='black', linewidth=1.5))  # Top-left to bottom-right
                                ax2.add_line(plt.Line2D([x_start, x_end], [y_end, y_start], color='black', linewidth=1.5))  # Bottom-left to top-right

                                # Collect x-axis labels that correspond to the "X"
                                dagger_labels.add(frame2.columns[x])
                    
                    # Modify x-axis labels to include daggers
                    x_labels = ax2.get_xticklabels()
                    new_labels = [
                        f"† {label.get_text()}" if label.get_text() in dagger_labels else label.get_text()
                        for label in x_labels
                    ]
                    ax2.set_xticklabels(new_labels, rotation=90, ha='right')
                    # Place the legend below the first heatmap
                    left2, right2 = ax1.get_xlim()
                    top2, bottom2 = ax1.get_ylim() 
                    ax2.text(left2, top2 + 7, 
                                '†: The NTC sample for this assay was removed from the analysis due to potential contamination.', 
                                ha='left', fontsize=12, style='italic')
                   
                
                # Adjust layout
                ax2.set_title(f'Heatmap for {barcode_number} at {time_assign[i]} minutes (Plate #2: {half_samples} Samples)', size=28)
                ax2.set_xlabel('Samples', size=14)
                ax2.set_ylabel('Assays', size=14)
                top, bottom = ax2.get_ylim() 
                ax2.set_ylim(top + 0.25, bottom - 0.25)
                left, right = ax2.get_xlim()
                ax2.set_xlim(left - 0.25, right + 0.25)
                ax2.tick_params(axis="y", labelsize=16)
                ax2.tick_params(axis="x", labelsize=16)
                plt.yticks(rotation=0)
                plt.tight_layout()
                ax2.axhline(y=top + 0.16, color='k',linewidth=6)
                ax2.axhline(y=bottom - 0.14, color='k',linewidth=6)
                ax2.axvline(x=left - 0.14, color='k',linewidth=6)
                ax2.axvline(x=right + 0.15, color='k',linewidth=6)
                
                # Save the figure to the dictionary
                fig_timepoints[i] = fig

        else: 
            for i in tqdm(tp):
                df_dict[i] = df_dict[i].transpose()

                # Do not split heatmap into two subplots (2-row, 1-column layout)
                fig, axes = plt.subplots(1, 1, figsize=(len(sample_list)*0.5,len(assay_list)*0.5)) # fig, axes = plt.subplots(1, 1, figsize=(len(sample_list)*0.5,len(sample_list)*0.5 * 2))
                # Add space between the two subplots (vertical spacing)
                plt.subplots_adjust(hspace=1)
                # add space to the bottom of the figure (adjust the bottom margin)
                plt.subplots_adjust(top=0.8, bottom=0.3)  

                # Plot heatmap (all samples)
                frame = df_dict[i][sample_list].reindex(assay_list)
                annot1 = frame.map(lambda x: 'X' if (pd.isna(x) or x == 'NaN' or x is None) else '')
                ax = sns.heatmap(frame, cmap='Reds', square=True, cbar_kws={'pad': 0.002}, annot = None, fmt='', annot_kws={"size": 1000, "color": "black"}, 
                                  linewidths = 1, linecolor = "black")
                # set colorbar format
                cbar = ax.collections[0].colorbar
                cbar.outline.set_edgecolor('black')  # Set the color of the edge (outline)
                cbar.outline.set_linewidth(2)  
                
                # calculate the real timing of the image
                rt = time_assign[i]
                
                # Track x-axis labels that need a dagger
                dagger_labels = set()
                
                # Add cross-hatches for "X"-marked cells
                if not annot1.empty: 
                    for y in range(annot1.shape[0]):
                        for x in range(annot1.shape[1]):
                            if annot1.iloc[y, x] == 'X':
                                # Calculate cell coordinates
                                x_start, y_start = x, y
                                x_end, y_end = x + 1, y + 1

                                # Add cross-hatches
                                ax.add_line(plt.Line2D([x_start, x_end], [y_start, y_end], color='black', linewidth=1.5))  # Top-left to bottom-right
                                ax.add_line(plt.Line2D([x_start, x_end], [y_end, y_start], color='black', linewidth=1.5))  # Bottom-left to top-right

                                # Collect x-axis labels that correspond to the "X"
                                dagger_labels.add(frame.columns[x])
                    
                    # Modify x-axis labels to include daggers
                    x_labels = ax.get_xticklabels()
                    new_labels = [
                        f"† {label.get_text()}" if label.get_text() in dagger_labels else label.get_text()
                        for label in x_labels
                    ]
                    ax.set_xticklabels(new_labels, rotation=90, ha='right')
                    # Place the legend below the first heatmap
                    left, right = ax.get_xlim()
                    top, bottom = ax.get_ylim() 
                    ax.text(left, top + 10, 
                                '†: The NTC sample for this assay was removed from the analysis due to potential contamination.', 
                                ha='left', fontsize=12, style='italic')
                               
                # Adjust layout
                ax.set_title(f'Heatmap for {barcode_number} at '+str(rt)+' minutes', size=28)
                ax.set_xlabel('Samples', size=18)
                ax.set_ylabel('Assays', size=18)
                top, bottom = ax.get_ylim() 
                ax.set_ylim(top + 0.25, bottom - 0.25)
                left, right = ax.get_xlim()
                ax.set_xlim(left - 0.25, right + 0.25)
                ax.tick_params(axis="y", labelsize=16, width = 2, length = 5)
                ax.tick_params(axis="x", labelsize=16, width = 2, length = 5)
                plt.yticks(rotation=0)
                plt.tight_layout()
                ax.axhline(y=top + 0.16, color='k',linewidth=6)
                ax.axhline(y=bottom - 0.14, color='k',linewidth=6)
                ax.axvline(x=left - 0.14, color='k',linewidth=6)
                ax.axvline(x=right + 0.15, color='k',linewidth=6)

                # Save the figure to the dictionary
                fig_timepoints[i] = fig

        return fig_timepoints
    
    