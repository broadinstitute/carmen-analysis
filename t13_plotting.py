import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import matplotlib.patches as patches

class t13_Plotter:
    def __init__(self):
        pass
    
    def gettime(time_assign, timepoints):
        realt = time_assign[timepoints]
        return (realt)

    def t13_plt_heatmap(self, tgap, barcode_number, df, sample_list, assay_list, tp, invalid_samples, invalid_assays, rnasep_df_heatmap):
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

                # Split heatmap into two subplots (2-row, 1-column layout)
                fig, axes = plt.subplots(2, 1, figsize=(len(first_half_samples) * 0.5, len(assay_list) * 0.5 * 2))
                # Add space between the two subplots (vertical spacing)
                plt.subplots_adjust(hspace=0.5)

                df = df.transpose()

                # First heatmap (first 96 samples)
                frame1 = df[first_half_samples].reindex(assay_list)
                # [assay.upper() for assay in invalid_assays]
                frame1.index = frame1.index.str.upper() 
                # assess if there are NTCs with NaN signal 
                annot1 = frame1.map(lambda x: 'X' if (pd.isna(x) or x == 'NaN' or x is None) else '')
                ax1 = sns.heatmap(frame1, cmap='Reds', square=True, cbar_kws={'pad': 0.002}, ax=axes[0], 
                                    linewidths = 1, linecolor = "black") # cbar_kws and pad adjusts the space between heatmap and the colorbar
                                    # annot = None, fmt='', annot_kws={"size": 1000, "color": "black"}
                
                # Track x-axis labels that need a dagger
                dagger_labels_1 = set()
                
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
                                dagger_labels_1.add(frame1.columns[x])
                    """
                    # Modify x-axis labels to include daggers
                    x_labels_1 = ax1.get_xticklabels()
                    new_labels_1 = [
                        f"† {label.get_text()}" if label.get_text() in dagger_labels_1 else label.get_text() for label in x_labels_1
                    ]
                    ax1.set_xticklabels(new_labels_1, rotation=90, ha='right')
                    """
                    # Place the legend below the first heatmap
                    left1, right1 = ax1.get_xlim()
                    top1, bottom1 = ax1.get_ylim() 
                    ax1.text(left1, top1 + 9, 
                                '†: The NTC sample for this assay was removed from the analysis due to potential contamination.', 
                                ha='left', fontsize=12, style='italic')

                # plot * on y-axis that contains Invalid Assays
                if invalid_assays:
                    invalid_assays = [assay.upper() for assay in invalid_assays]
                    asterisk1_labels = [label + '*' if label in invalid_assays else label for label in frame1.index]
                    ax1.set_yticklabels(asterisk1_labels, rotation=0)
                    
                    ## add legend for * below the '*: ...' legend
                    ax1.text(left1, top1 + 10, 
                                '*: This assay is considered invalid due to failing Quality Control Test #3, which evaluates performance of the Combined Positive Control sample.', 
                                ha='left', fontsize=12, style='italic')

                """  
                ## denoting rnasep neg samples adds no new information to the heatmap
                # plot ** on x-axis for samples that are negative for RNaseP
                if not rnasep_df_heatmap.empty:
                    # collect all unique samples from all cols
                    neg_rnasep_samples = pd.unique(rnasep_df_heatmap.values.ravel())
                    # remove NaN values
                    neg_rnasep_samples = [sample for sample in neg_rnasep_samples if pd.notna(sample)] 
                    # uppercase everything
                    neg_rnasep_samples = [sample.upper() for sample in neg_rnasep_samples]
                    frame1.columns = [col.upper() for col in frame1.columns]
                    #neg_rnasep_samples = [sample.strip().upper() for sample in neg_rnasep_samples]
                    #frame1.columns = [col.strip().upper() for col in frame1.columns]
                    # assign ** to neg rnasep samples

                    frame1_columns_stripped = [label.strip() for label in frame1.columns]
                    neg_rnasep_samples_stripped = [sample.strip() for sample in neg_rnasep_samples]

                    # Now create the list of labels with `**` appended for the negative samples
                    asterisk2_labels = [
                        f'**{label}' if label_stripped in neg_rnasep_samples_stripped else label
                        for label, label_stripped in zip(frame1.columns, frame1_columns_stripped)
                    ]

                    #asterisk2_labels = [f'{label}**' if label.strip() in neg_rnasep_samples else label for label in frame1.columns]
                    ax1.set_xticklabels(asterisk2_labels, rotation=90, ha='right')
                 
                    
                    ## add legend for * below the '*: ...' legend
                    ax1.text(left1, top1 + 11, 
                                '**: This sample is negative for human internal control, RNaseP. There are a few different implications of this result. See Quality Control Report for further explanation.', 
                                ha='left', fontsize=12, style='italic')
                """ 
                # plot *** on x-axis that contains Invalid Samples
                if invalid_samples.size > 0:
                    invalid_samples = [sample.upper() for sample in invalid_samples]
                    """
                    asterisk3_labels = [label + '***' if label in invalid_samples else label for label in frame1.columns]
                    ax1.set_xticklabels(asterisk3_labels, rotation=90, ha='right')
                    """
                    ## add legend for * below the '***: ...' legend
                    ax1.text(left1, top1 + 11, 
                                '***: This sample is invalid due to testing positive against the no-crRNA assay, an included negative assay control.', 
                                ha='left', fontsize=12, style='italic')

                 # plot new x-labels that combine asterisk and dagger logic
                x_labels_1 = ax1.get_xticklabels()
                final_labels = [
                        f"{'† ' if label.get_text().strip() in dagger_labels_1 else ''}"
                        f"{label.get_text().strip()}"
                        f"{'***' if label.get_text().strip().upper() in invalid_samples else ''}"
                        for label in x_labels_1
                    ]
                ax1.set_xticklabels(final_labels, rotation=90, ha='right')

                # fill in black box for any non-panel assays for panel-specific CPC samples 
                if all(('P1' in idx or 'P2' in idx or 'RVP' in idx) for idx in frame1.index) and all(('P1' in col or 'P2' in col or 'RVP' in col) for col in frame1.columns): 
                    for sample in frame1.columns:
                        if 'CPC' in sample:
                            sample_suffix = sample.split('_')[-1]
                            for assay in frame1.index:
                                if sample_suffix == 'RVP' and 'RVP' not in assay:
                                    x = frame1.columns.get_loc(sample)
                                    y = frame1.index.get_loc(assay) 
                                    #ax1.plot(x, y, 'ro')  # Plot red dots at (x, y)
                                    rect = patches.Rectangle((x, y), 1, 1, edgecolor='black', facecolor='black', fill=True, visible=True, zorder=100)
                                    ax1.add_patch(rect)
                                if sample_suffix == 'P1' and 'P1' not in assay:
                                    x = frame1.columns.get_loc(sample)
                                    y = frame1.index.get_loc(assay) 
                                    #ax1.plot(x, y, 'ro')  # Plot red dots at (x, y)
                                    rect = patches.Rectangle((x, y), 1, 1, edgecolor='black', facecolor='black', fill=True, visible=True, zorder=100)
                                    ax1.add_patch(rect)
                                if sample_suffix == 'P2' and 'P2' not in assay:
                                    x = frame1.columns.get_loc(sample)
                                    y = frame1.index.get_loc(assay) 
                                    #ax1.plot(x, y, 'ro')  # Plot red dots at (x, y)
                                    rect = patches.Rectangle((x, y), 1, 1, edgecolor='black', facecolor='black', fill=True, visible=True, zorder=100)
                                    ax1.add_patch(rect)
                
                # Adjust layout
                ax1.set_title(f'Heatmap for {barcode_number} at {time_assign[last_key]} minutes (Plate #1: {half_samples} Samples)', size=28)
                ax1.set_xlabel('Samples', size=18)
                ax1.set_ylabel('Assays', size=18)
                top1, bottom1 = ax1.get_ylim() 
                ax1.set_ylim(top1 + 0.25, bottom1 - 0.25)
                left1, right1 = ax1.get_xlim()
                ax1.set_xlim(left1 - 0.25, right1 + 0.25)
                ax1.tick_params(axis="y", labelsize=16, width = 2, length = 5)
                ax1.tick_params(axis="x", labelsize=16, width = 2, length = 5)
                plt.yticks(rotation=0)
                #plt.tight_layout(h_pad=0.3)
                ax1.axhline(y=top1 + 0.16, color='k',linewidth=6)
                ax1.axhline(y=bottom1 - 0.14, color='k',linewidth=6)
                ax1.axvline(x=left1 - 0.14, color='k',linewidth=6)
                ax1.axvline(x=right1 + 0.15, color='k',linewidth=6)

                # Second heatmap (next 96 samples)
                frame2 = df[second_half_samples].reindex(assay_list)
                frame2.index = frame2.index.str.upper() 
                # assess if there are NTCs with NaN signal 
                annot2 = frame2.map(lambda x: 'X' if (pd.isna(x) or x == 'NaN' or x is None) else '')
                ax2 = sns.heatmap(frame2, cmap='Reds', square=True, cbar_kws={'pad': 0.002}, ax=axes[1], 
                                    linewidths = 1, linecolor = "black") # annot = None, annot_kws={"size": 20}
                
                # Track x-axis labels that need a dagger
                dagger_labels_2 = set()
                
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
                                dagger_labels_2.add(frame2.columns[x])
                    
                    """  
                    # Modify x-axis labels to include daggers
                    x_labels_2 = ax2.get_xticklabels()
                    new_labels_2 = [
                        f"† {label.get_text()}" if label.get_text() in dagger_labels_2 else label.get_text() for label in x_labels_2
                    ]
                    ax2.set_xticklabels(new_labels_2, rotation=90, ha='right')
                    """
                    # Place the legend below the first heatmap
                    left2, right2 = ax2.get_xlim()
                    top2, bottom2 = ax2.get_ylim() 
                    ax2.text(left1, top2 + 9, 
                                '†: The NTC sample for this assay was removed from the analysis due to potential contamination.', 
                                ha='left', fontsize=12, style='italic')
                    
                # plot * on y-axis that contains Invalid Assays
                if invalid_assays:
                    invalid_assays = [assay.upper() for assay in invalid_assays]
                    asterisk1_labels = [label + '*' if label in invalid_assays else label for label in frame2.index]
                    ax2.set_yticklabels(asterisk1_labels, rotation=0)
                    
                    ## add legend for * below the '†: ...' legend
                    ax2.text(left1, top2 + 10, 
                                '*: This assay is considered invalid due to failing Quality Control Test #3, which evaluates performance of the Combined Positive Control sample.', 
                                ha='left', fontsize=12, style='italic')
                
                # plot *** on x-axis that contains Invalid Samples
                if invalid_samples:
                    invalid_samples = [sample.upper() for sample in invalid_samples]
                    """
                    asterisk3_labels = [label + '***' if label in invalid_samples else label for label in frame2.columns]
                    ax2.set_xticklabels(asterisk3_labels, rotation=90, ha='right')
                    """
                    ## add legend for * below the '†: ...' legend
                    ax2.text(left1, top2 + 11, 
                                '***: This sample is invalid due to testing positive against the no-crRNA assay, an included negative assay control.', 
                                ha='left', fontsize=12, style='italic')
                
                # plot new x-labels that combine asterisk and dagger logic
                x_labels_2 = ax2.get_xticklabels()
                final_labels = [
                        f"{'† ' if label.get_text().strip() in dagger_labels_2 else ''}"
                        f"{label.get_text().strip()}"
                        f"{'***' if label.get_text().strip().upper() in invalid_samples else ''}"
                        for label in x_labels_2
                    ]
                ax2.set_xticklabels(final_labels, rotation=90, ha='right')

                # fill in black box for any non-panel assays for panel-specific CPC samples 
                if all(('P1' in idx or 'P2' in idx or 'RVP' in idx) for idx in frame2.index) and all(('P1' in col or 'P2' in col or 'RVP' in col) for col in frame2.columns): 
                    for sample in frame2.columns:
                        if 'CPC' in sample:
                            sample_suffix = sample.split('_')[-1]
                            for assay in frame2.index:
                                if sample_suffix == 'RVP' and 'RVP' not in assay:
                                    x = frame2.columns.get_loc(sample)
                                    y = frame2.index.get_loc(assay) 
                                    #ax2.plot(x, y, 'ro')  # Plot red dots at (x, y)
                                    rect = patches.Rectangle((x, y), 1, 1, edgecolor='black', facecolor='black', fill=True, visible=True, zorder=100)
                                    ax2.add_patch(rect)
                                if sample_suffix == 'P1' and 'P1' not in assay:
                                    x = frame2.columns.get_loc(sample)
                                    y = frame2.index.get_loc(assay) 
                                    #ax2.plot(x, y, 'ro')  # Plot red dots at (x, y)
                                    rect = patches.Rectangle((x, y), 1, 1, edgecolor='black', facecolor='black', fill=True, visible=True, zorder=100)
                                    ax2.add_patch(rect)
                                if sample_suffix == 'P2' and 'P2' not in assay:
                                    x = frame2.columns.get_loc(sample)
                                    y = frame2.index.get_loc(assay) 
                                    #ax2.plot(x, y, 'ro')  # Plot red dots at (x, y)
                                    rect = patches.Rectangle((x, y), 1, 1, edgecolor='black', facecolor='black', fill=True, visible=True, zorder=100)
                                    ax2.add_patch(rect)

                # Adjust layout
                ax2.set_title(f'Heatmap for {barcode_number} at {time_assign[last_key]} minutes (Plate #2: {half_samples} Samples)', size=28)
                ax2.set_xlabel('Samples', size=18)
                ax2.set_ylabel('Assays', size=18)
                top, bottom = ax2.get_ylim() 
                ax2.set_ylim(top + 0.25, bottom - 0.25)
                left, right = ax2.get_xlim()
                ax2.set_xlim(left - 0.25, right + 0.25)
                ax2.tick_params(axis="y", labelsize=16)
                ax2.tick_params(axis="x", labelsize=16)
                plt.yticks(rotation=0)
                #plt.tight_layout(h_pad=0.3)
                ax2.axhline(y=top + 0.16, color='k',linewidth=6)
                ax2.axhline(y=bottom - 0.14, color='k',linewidth=6)
                ax2.axvline(x=left - 0.14, color='k',linewidth=6)
                ax2.axvline(x=right + 0.15, color='k',linewidth=6)

                fig.tight_layout()

            else: 
                # Do not split heatmap into two subplots (2-row, 1-column layout)
                fig, axes = plt.subplots(1, 1, figsize=(len(frame.columns.values)*0.5,len(frame.index.values)*0.5 * 2))
                # Add space between the two subplots (vertical spacing)
                plt.subplots_adjust(hspace=1)

                # Plot heatmap (all samples)
                df = df.transpose()
                frame = df[sample_list].reindex(assay_list)
                annot1 = frame.map(lambda x: 'X' if (pd.isna(x) or x == 'NaN' or x is None) else '')
                ax = sns.heatmap(frame, cmap='Reds', square=True, cbar_kws={'pad': 0.002}, annot = None, fmt='', annot_kws={"size": 1000, "color": "black"}, ax=axes[0], 
                                    linewidths = 1, linecolor = "black")
                
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
                    ax.text(left, top + 7, 
                                '†: The NTC sample for this assay was removed from the analysis due to potential contamination.', 
                                ha='left', fontsize=12, style='italic')
                
                # plot * on y-axis that contains Invalid Assays
                if invalid_assays:
                    invalid_assays = [assay.upper() for assay in invalid_assays]
                    asterisk1_labels = [label + '*' if label in invalid_assays else label for label in frame1.index]
                    ax.set_yticklabels(asterisk1_labels, rotation=0)
                    
                    ## add legend for * below the '†: ...' legend
                    ax.text(left1, top1 + 9, 
                                '*: This assay is considered invalid due to failing Quality Control Test #3, which evaluates performance of the Combined Positive Control sample.', 
                                ha='left', fontsize=12, style='italic')
                
                # plot *** on x-axis that contains Invalid Samples
                if invalid_samples.size > 0:
                    invalid_samples = [sample.upper() for sample in invalid_samples]
                    asterisk3_labels = [label + '***' if label in invalid_samples else label for label in frame1.columns]
                    ax.set_xticklabels(asterisk3_labels, rotation=90, ha='right')

                    ## add legend for * below the '†: ...' legend
                    ax.text(left1, top1 + 10, 
                                '***: This sample is invalid due to testing positive against the no-crRNA assay, an included negative assay control.', 
                                ha='left', fontsize=12, style='italic')

                # fill in black box for any non-panel assays for panel-specific CPC samples 
                if all(('P1' in idx or 'P2' in idx or 'RVP' in idx) for idx in frame.index) and all(('P1' in col or 'P2' in col or 'RVP' in col) for col in frame.columns): 
                    for sample in frame.columns:
                        if 'CPC' in sample:
                            sample_suffix = sample.split('_')[-1]
                            for assay in frame.index:
                                if sample_suffix == 'RVP' and 'RVP' not in assay:
                                    x = frame.columns.get_loc(sample)
                                    y = frame.index.get_loc(assay) 
                                    #ax1.plot(x, y, 'ro')  # Plot red dots at (x, y)
                                    rect = patches.Rectangle((x, y), 1, 1, edgecolor='black', facecolor='black', fill=True, visible=True, zorder=100)
                                    ax.add_patch(rect)
                                if sample_suffix == 'P1' and 'P1' not in assay:
                                    x = frame.columns.get_loc(sample)
                                    y = frame.index.get_loc(assay) 
                                    #ax1.plot(x, y, 'ro')  # Plot red dots at (x, y)
                                    rect = patches.Rectangle((x, y), 1, 1, edgecolor='black', facecolor='black', fill=True, visible=True, zorder=100)
                                    ax.add_patch(rect)
                                if sample_suffix == 'P2' and 'P2' not in assay:
                                    x = frame.columns.get_loc(sample)
                                    y = frame.index.get_loc(assay) 
                                    #ax1.plot(x, y, 'ro')  # Plot red dots at (x, y)
                                    rect = patches.Rectangle((x, y), 1, 1, edgecolor='black', facecolor='black', fill=True, visible=True, zorder=100)
                                    ax.add_patch(rect)
                    
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
        
            return fig