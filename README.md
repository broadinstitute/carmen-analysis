# Welcome! 
CARMEN is a diagnostic tool designed for surveillance purposes. Below are the instructions to complete your CARMEN analysis. 

## Overview
At this point, you have ran the IFC (integrated fluidic circuit) on the Standard BioTools BioMark instrument and have completed the experimental portion of CARMEN. In running this code, you will be able to complete the data analysis portion of CARMEN and generate both binary positive/negative and quantitative signal output of your diagnostic assay. 

This code is pathogen-agnostic and can be utilized for any combination of viral assays ran on the Standard BioTools IFC used in CARMEN. The contents of the output folder are described below, with reference to experimental analysis performed for a 192.24 IFC chip and a 1-hour Standard Biotools Biomark run.

## Primer on the structure of your cloned working directory and the virtual environment
The following is a brief discussion fundamental structure of your cloned directory and the virtual environment. 

Let's say you have the working directory ``\Users\albeez\Sentinel``. In the Sentinel directory, I have the following sub-directories (or sub-folders): ``\Users\albeez\Sentinel\CARMEN_Run_1``, ``\Users\albeez\Sentinel\CARMEN_Run_2``, and ``\Users\albeez\Sentinel\CARMEN_Run_3``. 

You are currently in the process of analyzing ``CARMEN_Run_2``. Thus, from the terminal, you will navigate to the location ``\Users\albeez\Sentinel\CARMEN_Run_2`` using the ``cd`` command. Once in the ``CARMEN_Run_2`` directory, you will clone the Github repository as per the instructions below (specifically in Step 3). 

The cloned directory's path will be ``\Users\albeez\Sentinel\CARMEN_Run_2\carmen-sentinel-analysis``. 

Next, from the terminal, you will create a virtual environment in the cloned directory ``\carmen-sentinel-analysis`` as per the instructions below (specifically in Step 4). 

Of note, if you need to work in this working directory (``\Users\albeez\Sentinel\CARMEN_Run_2\carmen-sentinel-analysis``) again, you will not need to create the virtual environment again. You will only have to activate the virtual environment and install the dependencies.

In summary, the cloned directory contains all of the needed python scripts, data files, and the virtual environment. The virtual environment is separate from the Python files and data files - but it will access the files in the cloned directory. 

## Installation
Now that you have a high-level understanding of the directories you need to create, let's start the installation process. 

1. Open Terminal on your computer. 

2. Change the current working directory to the location where you want the cloned directory. (Use the command ``cd`` to change the path and use the command `ls` to show what files you have available in the working directory.) It is recommended that 

3. Clone the repository. 
From your terminal, type ``git clone https://github.com/cwilkason/carmen-streamlit-v1``

4. Launch a virtual environment in your cloned directory. Follow the delineated steps to install needed dependencies:
* Run the following command from your terminal to create a virtual environment in your cloned directory: ``python -m venv carmen-env``
* Run the following command from your terminal to activate the virtual environment by running the following command in your terminal: ``source ./carmen-env/bin/activate``
* Run the following command from your terminal to install required dependencies in the virtual environment: ``pip3 install -r requirements.txt``

Note: When working in this cloned directory in the future, you will not need to create the virtual environment again. You will only have to activate the virtual environment and install the dependencies.


### Required file inputs 
You will need the following two files to complete the CARMEN analysis:
* A .xlsx Assignment Sheet. Enter your samples and assays in the corresponding tabs in the .xlsx template file provided. Do not edit the layout_assays and layout_samples tabs in the .xlsx template file provided. Rename this file as follows: {IFC Barcode}_192_assignment.xlsx. NOTE: Your negative control must contain the phrase "NTC".
* A .csv Data File from the Standard BioTools instrument. When exporting results from teh Standard Biotools instrument, this output file is called Results_all.csv. Rename this file as follows: {IFC_Barcode}.csv

1. Get the path of your current working directory by running the following command in the terminal: ``pwd``
2. Copy this path
3. Return to your home directory by running the following command in the terminal: ``cd ~`` 
4. Navigate to the location of your Assignment Sheet (.xlsx) using the ``cd`` command in the terminal. 
5. When you are in the directory containing the Assignment Sheet (.xlsx), run the following command in the terminal: ``mv {IFC Barcode}_192_assignment.xlsx /pathToClonedDirectory``
* Replace ``{IFC Barcode}_192_assignment.xlsx`` with the name of your Assignment Sheet 
* In place of ``/pathToClonedDirectory``, paste in the Cloned Directory path you copied 
6. Repeat steps 3, 4, and 5 to move your Data File (.csv) to the Cloned Directory. 

NOTE: Remove all other .xlsx files from the current working directory by running the following command from your terminal: 
1. Check if there are any other .xlsx files in the current working directory by running the following command in the terminal: ``ls *.xlsx``
2. If there are other .xlsx files in the current working directory, remove them by running the following command in the terminal ``find . -name '*.xlsx' ! -name '192_assignment.xlsx' -exec rm {} \;`` or move them to another location on your computer by running this command in the terminal ``find . -name '*.xlsx' ! -name '192_assignment.xlsx' -exec mv {} /pathToAnotherFolder \;``

NOTE: Remove all other .csv files from the current working directory by running the following command from your terminal: 
3. Check if there are any other .csv files in the current working directory by running the following command in the terminal: ``ls *.csv``
4. If there are other .csv files in the current working directory, remove them by running the following command in the terminal ``find . -name '*.xlsx' ! -name '{IFC Barcode}.csv' -exec rm {} \;`` or move them to another location on your computer by running this command in the terminal ``find . -name '*.xlsx' ! -name '{IFC Barcode}.csv' -exec mv {} /pathToAnotherFolder \;``


## Usage
To run the analysis, type the following command in your terminal: ``python3 carmen_analysis.py ``. The output of this command will be a folder named ``output_{IFC Barcode}`` located in your cloned directory. 

For the ``CARMEN_Run_2`` example given above, the output folder's path would be ``\Users\albeez\Sentinel\CARMEN_Run_2\carmen-sentinel-analysis\CARMEN_Run_2``. 

## Description of the Outputs
After successfully running this code, you will produce a folder called "output" containing 21 csv files and 13 figures (assuming you are running a 192.24 IFC chip). If your primary purpose is diagnostic surveillance, the files most useful for you will be as follows: 
* t13_{IFC Barcode}.csv : Quantitative signal output
* t13_{IFC Barcode}_hit_output.csv : Positive/Negative binary output
* t13_{IFC Barcode}_quant_ntcNorm.csv : Normalized quantitative signal output wherein the signal per assay and per sample has been normalized against the mean NTC (No Target Control) signal. 
* Heatmap_t13_{IFC Barcode}.png : Heatmap for data visualization at the final timepoint.
* Positives_Summary_{IFC Barcode}.csv : Summary of total positive samples per assay and tabulation of the sample ID of each positive sample per assay. 

If you are more interested in research & development in the diagnostic space, additional files you might be interested are as follows:
* t{#}_{IFC Barcode}.csv : These are the quantitative signal output per timepoint. (For a 1-hour Standard Biotools Biomark run, you will generate 13 such total files corresponding to 13 timepoints.)
* Heatmap_t{#}_{IFC Barcode}.png : Heatmap for data visualization  per timepoint. (For a 1-hour Standard Biotools Biomark run, you will generate 13 such total files corresponding to 13 timepoints.)
* ref_norm.csv : This file corresponds to the signal of the passive reference ROX dye per timepoint, per assay used in your experiment.
* assigned_ref_norm.csv: This is a more useful version of the ref_norm.csv as it shows you which sample and which assay the passive reference ROX dye corresponds to, at each timepoint.
* signal_norm.csv : This file corresponds to the normalized FAM signal (after subtraction of the background FAM signal and normalization against the signal from the reference dye).
* assigned_signal_norm.csv : This is a more useful version of the signal_norm.csv as it shows you which sample and which assay the normalized FAM signal corresponds to, at each timepoint.
* NTC_thresholds_{IFC Barcode}.csv : This file tabulates the mean NTC value thresholds at which samples were 


## Congratulations on analyzing your CARMEN run! 
If you would like to learn more about the contents of the cloned directory, read the description provided below. 


## Content Description: of Python files
The cloned directory contains the following python scripts:
* carmen_analysis.py
* matcher.py
* median_frame.py
* norm.py
* ntcnorm.py
* reader.py
* threshold.py 

``threshold.py``
This file does the thresholding to evaluate what is a positive output 

``ntcnorm.py``
This file performs normalization on the NTC values to produce the NTC_quant_norm output. Take the NTC mean per assay and divide the raw signal for all samples for that assay by the NTC mean. 



## Questions
If you have any questions or run into any questions regarding the code base, please reach out to albeez@broadinstitute.org. 

