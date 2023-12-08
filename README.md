# Welcome! 
Below are the instructions to complete your CARMEN analysis. 

## Overview
At this point, you have ran the IFC (integrated fluidic circuit) on the Standard BioTools BioMark instrument and have completed the experimental portion of CARMEN. In running this code, you will be able to complete the data analysis portion of CARMEN and generate both binary positive/negative and quantitative signal output of your diagnostic assay. 

This code is pathogen-agnostic and can be utilized for any combination of viral assays ran on the Standard BioTools IFC used in CARMEN. 

After successfully running this code, you will produce a folder called "output" containing 19 csv files. If your primary purpose is diagnostic surveillance, the files most useful for you will be as follows: 
* t13_{IFC Barcode}.csv : Quantitative signal output
* t13_{IFC Barcode}_hit_output.csv : Positive/Negative binary output
* t13_{IFC Barcode}_quant_hit_output_ntcNorm.csv : Normalized quantitative signal output wherein the signal per assay and per sample has been normalized against the mean NTC (No Target Control) signal. 

If you are more interested in research & development in the diagnostic space, additional files you might be interested are as follows:
* t{#}_{IFC Barcode}.csv : These are the quantitative signal output per timepoint. (For a 1-hour Standard Biotools Biomark run, you will generate 13 such total files corresponding to 13 timepoints.)
* ref_norm.csv : This file corresponds to the signal of the passive reference ROX dye per timepoint, per assay used in your experiment.
* assigned_ref_norm.csv: This is a more useful version of the ref_norm.csv as it shows you which sample and which assay the passive reference ROX dye corresponds to, at each timepoint.
* signal_norm.csv : This file corresponds to the normalized FAM signal (after subtraction of the background FAM signal and normalization against the signal from the reference dye).
* assigned_signal_norm.csv : This is a more useful version of the signal_norm.csv as it shows you which sample and which assay the normalized FAM signal corresponds to, at each timepoint.

## Prerequisities
* Python 3.x  installed

## Installation
1. Clone the repository. 
From your terminal, type ``git clone (Insert github url)``

2. Navigate to the project directory. 
Use --cd to navigate to the directory containing the python scripts for this folder. 

3. Install dependencies inside a virtual environment.
```
python -m venv carmen-env
source ./carmen-env/bin/activate
pip3 install -r requirements.txt
```


In your terminal, run ``pip3 install -r requirements.txt``

To run this code, clone this repository and save it to a folder on your computer. Open the terminal on your computer or alternate command line interface. Using the ls and cd commands, access the downloaded folder in your terminal. --ls will show you what files are in your path and --cd will allow you to go to the file in your path. Create a path to the downloaded repo. This is your working directory. 

### Required file inputs 
Make sure the current working directory contains the following python scripts:
* carmen_analysis.py
* marcher.py
* median_frame.py
* norm.py
* ntcnorm.py
* reader.py
* threshold.py 

Add the following files to the current working directory:
* A .xlsx Assignment Sheet. Enter your samples and assays in the corresponding tabs in the .xlsx template file provided. Do not edit the layout_assays and layout_samples tabs in the .xlsx template file provided. Rename this file as follows: {IFC Barcode}_192_assignment.xlsx 
* A .csv Data File from the Standard BioTools instrument. When exporting results from teh Standard Biotools instrument, this output file is called Results_all.csv. Rename this file as follows: {IFC_Barcode}.csv

Remove all other .xlsx and .csv files from the current working directory.

## Usage
``python3 carmen_analysis.py ``

## Questions
If you have any questions or run into any questions regarding the code base, please reach out to albeez@broadinstitute.org. 

