# Welcome! 
CARMEN is a diagnostic tool designed for surveillance purposes. Below are the instructions to complete your CARMEN analysis. 

## Software Version
When cloning this repository, you will be using software version 5.4.2.

## Overview
At this point, you have ran the $Standard\ BioTools\ Dynamic\ Array^{TM}$ IFC (integrated fluidic circuit) on the $Standard\ BioTools\ Biomark^{TM}$ instrument and have completed the experimental portion of CARMEN. In running this code, you will be able to complete the data analysis portion of CARMEN and generate both binary positive/negative and quantitative signal output of your diagnostic assay. 

This code is pathogen-agnostic and can be utilized for any combination of viral assays ran on the Standard BioTools IFC used in CARMEN. The contents of the output folder are described below, with reference to experimental analysis performed for a 192.24 IFC chip and a 1-hour Standard Biotools Biomark run.

## Table of Contents

[Primer on Cloning & the Python Virtual Environment](https://github.com/cwilkason/carmen-streamlit-v1?tab=readme-ov-file#primer-on-the-structure-of-your-cloned-working-directory-and-the-virtual-environment)

[Installation](https://github.com/cwilkason/carmen-streamlit-v1?tab=readme-ov-file#installation)

[Content Description of Cloned Working Directory](https://github.com/cwilkason/carmen-streamlit-v1?tab=readme-ov-file#content-description-of-cloned-directory)

[Required File Inputs](https://github.com/cwilkason/carmen-streamlit-v1?tab=readme-ov-file#required-file-inputs)
* [An .xlsx Assignment Sheet](https://github.com/cwilkason/carmen-streamlit-v1?tab=readme-ov-file#an-xlsx-assignment-sheet)
* [A .csv Data File from the Standard BioTools instrument](https://github.com/cwilkason/carmen-streamlit-v1?tab=readme-ov-file#a-csv-data-file-from-the-standard-biotools-instrument)

[File Relocation to Cloned Directory](https://github.com/cwilkason/carmen-streamlit-v1?tab=readme-ov-file#file-relocation-to-cloned-directory)

[Usage and Running the Analysis](https://github.com/cwilkason/carmen-streamlit-v1?tab=readme-ov-file#usage-and-running-the-analysis)

[Description of the Outputs](https://github.com/cwilkason/carmen-streamlit-v1?tab=readme-ov-file#description-of-the-outputs)

[Any Questions?](https://github.com/cwilkason/carmen-streamlit-v1?tab=readme-ov-file#congratulations-on-analyzing-your-carmen-run)

## Primer on Cloning & the Python Virtual Environment

The following is a brief discussion of the fundamental structure of your cloned directory and the virtual environment. 

Firstly, what is a working directory? Essentially, a working directory is a file path that points to where your program or file of interest is located on your computer. Let us assume that you have the working directory ``\Users\albeez\Sentinel``. This shows the file path on your computer to get to the folder called "Sentinel". 

In the Sentinel directory, let us assume that you have the following sub-directories (or sub-folders): ``\Users\albeez\Sentinel\CARMEN_Run_1``, ``\Users\albeez\Sentinel\CARMEN_Run_2``, and ``\Users\albeez\Sentinel\CARMEN_Run_3``. You are currently in the process of analyzing ``CARMEN_Run_2``. 

Thus, from the terminal, you will navigate to the location ``\Users\albeez\Sentinel\CARMEN_Run_2`` using the ``cd`` command. Once in the ``CARMEN_Run_2`` directory, you will clone the Github repository as per the instructions below (specifically in Step 3). The cloned directory's path will be ``\Users\albeez\Sentinel\CARMEN_Run_2\carmen-analysis``. 

Next, from the terminal, you will create a virtual environment in the cloned directory ``\carmen-analysis`` as per the instructions below (specifically in Step 4). 
  * **Note:** If you need to work in this working directory (``\Users\albeez\Sentinel\CARMEN_Run_2\carmen-analysis``) again, you will not need to create the virtual environment again. You will only have to activate the virtual environment and install the dependencies. 

In summary, the cloned directory contains all of the needed python scripts, data files, and the virtual environment. The virtual environment is separate from the Python files and data files - but it will access the files in the cloned directory. 

## Installation
Now that you have a high-level understanding of the directories you need to create, let's start the installation process. 

1. Open Power Shell with Administrator access (`Windows`) or Terminal (`macOS`) on your computer. 


2. Set the current working directory to the location where you want the cloned directory.

    * There are a few differences between the operating systems in regard to how to run this code. The instructions will detail such differences in detail. Please note the specific operating system running on your local machine. 
    * `macOS` and `Linux`: 
      * Use the command ``cd`` to change the path. 
      * Use the command `ls` to show what files you have available in sub-folders along the file path.
    * `Windows`: 
      * Use the command ``cd`` to change the path.
      * Use the command `dir` to show what files you have available in sub-folders along the file path.

3. Check if you have Python installed on your local machine by running ``python3 --version``. If you do not have Python on your local machine already, follow the instructions below to install Python.

    * `macOS`: 
      * You can download the latest version of Python [here](https://www.python.org/downloads/) and follow the package installer instructions.
      * After completing the installation, run ``python3 --version`` to verify successful instllation of Python.
    * `Ubuntu Linux`: 
      * For systems with Ubunto 16.10 installed, run the following commands to install Python:
        * ``sudo apt-get update``
        * ``sudo apt-get install python3``
    * `Windows`: 
      * You can download the latest version of Python [here](https://www.python.org/downloads/windows/) and follow the package installer instructions
        * **Note:** When the installer prompts you to do so, check the box to add Python to PATH. 
      * After completing the installation, run ``py --version`` to verify successful instllation of Python.

4. If your computer are using a ``macOS`` system, you need to have the Xcode Command Line Tools installed. To check if you already have installed the Xcode Command Line Tools on your macOs system, run the following command: ``xcode-select --version``. 

    * If you have not installed the Xcode Command Line Tools, you will get an error message as follows: ``xcode-select: note: no developer tools were found at '/Applications/Xcode.app', requesting install. Choose an option in the dialog to download the command line developer tools.`` To install the Xcode Command Line Tools, run the following command in your terminal: ``xcode-select --install``.


5. Check if you have `pip` installed by running the command: ``pip --version``

  * If you do not have `pip` installed, install it following these steps: 
      * `macOs`, `Linux`, and `Windows`: 
        * Download the get-pip.py file by running this command: ``curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py``
        * Install pip by running the command below:
          * `macOS` or `Linux`:  ``python get-pip.py``
          * `Windows`: ``py get-pip.py``
              * Run `pip --version` again.
              * If you get an error after installing `pip` on  Windows PC, you may have to set the PATH variable: `set PATH=%PATH%;C\Users\(file path)` 
              * The file path is indicated in the error (but should be the file path to the folder you are currently working in via the terminal).


6. Check if you have `git` installed by running the command: ``git --version``

  * If you do not have `git` installed, install it following these steps:
    * `macOS`: 
        * If you have installed the Xcode Command Line Tools described above, `git` should already be installed.
    * `Linux`: 
        * Depending on the Linux distribution you are using (Debian, Ubuntu, Fedora, etc), the installation will differ. 
        * Check [here](https://git-scm.com/download/linux) for specific installation instructions for `git`.
    * `Windows`: 
        * Download the `git` package [here] (https://git-scm.com/download/win)
        * Install `git` by running this command: `winget install --id Git.Git -e --source winget`
        * Ensure that `git` has been added to your system's PATH. Navigate to `Advanced system settings` > ` Environment Variables` > `System variables` > `Path`. 
        * Click the `Edit` button and if it is not there already, add `C:\Program Files\Git\bin` and `C:\Program Files (x86)\Git\bin`. Save the changes. 


7. Clone the repository. 
From your terminal, type ``git clone https://github.com/broadinstitute/carmen-analysis``

## Content Description of Cloned Directory 
The cloned directory should contain ALL of the following python scripts for your analysis to be successfully completed:
* ``analyze_run.py``
* ``assay_qc_score.py``
* ``binary_results.py``
* ``flags.py``
* ``matcher.py``
* ``median_frame.py``
* ``norm.py``
* ``ntc_con_check.py``
* ``ntcnorm.py``
* ``plotting.py``
* ``qual_checks.py``
* ``reader.py``
* ``requirements.txt``
* ``summary.py``
* ``t13_plotting.py``
* ``threshold.py`` 
* ``Assay-Level QC Test Explanation.pdf``
* ``LICENSE``
* ``README.md``

## Required File Inputs 
You will need the following two files to complete the CARMEN analysis.

### (1) An .xlsx Assignment Sheet
Go to [this Google Drive folder](https://drive.google.com/drive/folders/1iQsmyuwRtDyMgtT2YvgJ4yv_sGcJMhgv?usp=drive_link). Select the Assignment Sheet template  corresponding to the dimensions of the IFC chip you ran. Download the template as an ``.xlsx`` file.

See the file labelled ``SAMPLE 192.24 Assignment Sheet.xlsx`` as an example of an assignment sheet which contains the required controls and panel-specific formating described below.

Enter your samples in Sheet 1 and your assays Sheet 2 of the downloaded ``.xlsx`` template file. Do **NOT** edit the layout_assays and layout_samples tabs in the .xlsx template file provided. 

Rename this file as follows: ``{IFC Barcode}_{Chip Dimension}_assignment.xlsx`` 

#### (i) What does ``"{Chip Dimension}"`` mean?
``"{Chip Dimension}"`` refers to the total number of samples and number of assays that your $Standard\ BioTools\ Dynamic\ Array^{TM}$ IFC ran. 

The most common IFC chip ran in CARMEN is the ``192.24`` IFC chip, which runs 192 samples and 24 assays. However, there is also the ``96.96`` IFC chip, which runs 96 samples and 96 assays, and the ``48.48`` IFC chip, which runs 48 samples and 48 assays.

#### (ii) Required Controls
Each plate of samples that you run in CARMEN must include the following 3 controls:  
  1. First Negative control must contain the phrase ``"NTC"``.
    * The ``"No Target Control"`` does not contain viral target and should produce a negative response. 
    * Validation is integral to ensuring that there is no contamination in the RT-PCR steps of CARMEN. 
  2. Second Negative control must contrain the phrase ``"NDC"``. 
    * The ``"No Detection Control"`` does not contain Magnesium 2+ and should produce a negative response. 
    * Validation is integral to ensuring that there is no conamination in the Detection steps of CARMEN (preparing the Sample Master Mix and loading the chip). Cas13a enzyme remains unactive without its cofactor Magnesium 2+. 
  3. Positive control must contain the phrase ``"CPC"``. 
    * The ``"Combined Positive Control"`` contains a combined assay of synthetic targets corresponding to all viral assays tested in your CARMEN run. 
    * A positive signal for the CPC per viral assay is integral to validating the viral assay in your CARMEN run. 
  4. Negative Viral Assay Control must contain the phrase ``no_crRNA``. 
    * The ``no_crRNA`` control does not contain any viral crRNA assay and each sample should test negative against this assay.
    * A negative signal for the ``no_crRNA`` control is integral to validating the sample tested in your CARMEN run.

The assumption for the remainder of the CARMEN analysis is that any sample that does contain the phrases ``NTC``, ``NDC``, and ``CPC`` is considered a “patient or clinical sample”.

#### (iii) Panel-Specific Formating
Based on the viral panel you run in CARMEN, add the corresponding '_RVP', '_P1', or '_P2' as the suffix to samples and assays corresponding to RVP Panel, BBP Panel #1, or BBP Panel #2, respectively. 
* Note: If you have a control (for example, the RNaseP assay) that corresponds to two or more panels (for example, BBP Panel #1 and BBP Panel #2), you can denote the suffix of BOTH panels in the assay name (for example, 'RNaseP_P1_P2').

### (2) A .csv Data File from the Standard BioTools instrument
When exporting results from the Standard Biotools instrument, this output file is called Results_all.csv. Rename this file as follows: ``{IFC_Barcode}.csv``

## File Relocation to Cloned Directory
Move the required file inputs from their current locations on your computer to the cloned directory by using File Explorer or Finder. (If you are a user who is highly familiar with the command line, you can also achieve the same file relocation using only the terminal interface.)

1. Using Finder or File Explorer your local computer's, locate where you have saved the two required file inputs: {IFC Barcode}_192_assignment.xlsx and {IFC_Barcode}.csv
2. Move both of these files into the Cloned Directory (via a "drag and drop" method).
3. Remove all other .csv files from the Cloned Directory. 
4. Remove all other .csv files from the current working directory

For the ``CARMEN_Run_2`` example given above, the file path for the .xlsx Assignment Sheet will be ``\Users\albeez\Sentinel\CARMEN_Run_2\carmen-analysis\{IFC Barcode}_192_assignment.xlsx``

For the ``CARMEN_Run_2`` example given above, the file path for the .csv Data File will be ``\Users\albeez\Sentinel\CARMEN_Run_2\carmen-analysis\{IFC_Barcode}.csv``

## Usage and Running the Analysis
Launch a virtual environment **inside your cloned directory**. Follow the delineated steps to install needed dependencies:
1. From the command-line interface, enter inside the cloned working directory. For the ``CARMEN_Run_2`` example given above, the file path from which you perform the steps below should be ``\Users\albeez\Sentinel\CARMEN_Run_2\carmen-analysis``

2. Run the following command from your terminal to **create** a virtual environment in your cloned directory: 

    * `macOS` or `Linux`:  ``python3 -m venv carmen-env``
    * `Windows`: ``py -m venv carmen-env``

3. Run the following command from your terminal to **activate** the virtual environment by running the following command in your terminal: 
    * `macOS` or `Linux`: 
        * ``conda deactivate``
        * ``source ./carmen-env/bin/activate``
    * `Windows`: 
        * ``.\carmen-env\Scripts\Activate.ps1``
          * When you first try activating the virtual environment, you may get an error about `Execution_Policies`.
          * If so, run `Get-ExecutionPolicy` and then run `Set-ExecutionPolicy Unrestricted`.
          * Then proceed to activating the virtual environment.  

  When the virtual environment has been activated, the text ``(carmen-env)`` will be to the left-most of your command-line in the terminal. 

4. Run the following command from your terminal to install required dependencies in the virtual environment: ``pip3 install -r requirements.txt``

    * For the ``CARMEN_Run_2`` example given above, the virtual environment's file path would be ``\Users\albeez\Sentinel\CARMEN_Run_2\carmen-analysis\carmen-env``. 

    * **Note:** When working in this cloned directory in the future, you will not need to create the virtual environment again. You will only have to activate the virtual environment and install the dependencies.

5. To analyze, run the following command in your terminal: 

  * `macOS` or `Linux`: ``python3 analyze_run.py {CLI}``. Read below to understand what you should input for ``{CLI}``. 
  * `Windows`: ``py analyze_run.py {CLI}``. Read below to understand what you should input for ``{CLI}``. 
    
    CARMEN has been optimized for multiple viral assay panels. Some of these panels require different types of thresholding for determining positive vs negative samples. 

    * If you are running the Respiratory Virus Panel (RVP), or require your positive samples to be thresholded above **``1.8 * (the mean signal of the No Target Control samples)``**, then the command-line interface argument (CLI) you will need to enter in Step 5 is: ``1.8_Mean``.

    * If you are running the Blood-Borne Pathogens Panel (BBP), or require your positive samples to be thresholded **``3 times above the standard deviation of the mean No Target Control samples``**, then the command-line interface argument (CLI) you will need to enter in Step 5 is: ``3_SD``.


6. The output of the command ``python3 analyze_run.py {CLI}`` (`macOS`/ `Linux`) or ``py analyze_run.py {CLI}`` (`Windows`) will be a folder named ``output_{IFC Barcode}`` located in your cloned directory. 

    * For the ``CARMEN_Run_2`` example given above, the output folder's file path would be ``\Users\albeez\Sentinel\CARMEN_Run_2\carmen-analysis\output_{IFC Barcode}``. 

## Description of the Outputs
After successfully running this code, you will produce a folder called "output" containing 21 csv files and 13 figures (assuming you are running a standard 1 hour protocol on the $Standard\ BioTools\ Biomark^{TM}$ instrument).

 If your primary purpose is diagnostic surveillance, the files most useful for you will be as follows: 
* ``t13_{IFC Barcode}.csv`` : Quantitative signal output
* ``t13_{IFC Barcode}_hit_output.csv`` : Positive/Negative binary output
* ``t13_{IFC Barcode}_quant_ntcNorm.csv`` : Normalized quantitative signal output wherein the signal per assay and per sample has been normalized against the mean NTC (No Target Control) signal. 
* ``Heatmap_t13_{IFC Barcode}.png`` : Heatmap for data visualization at the final timepoint.
* ``Positives_Summary_{IFC Barcode}.csv`` : Summary of total positive samples per assay and tabulation of the sample ID of each positive sample per assay. 
* ``Quality_Control_Flags_{IFC Barcode}.csv`` : Summary of four quality control checks done on the analyzed results to give you a brief overview on the validation of the samples and assays tested in this experiment. The report consists of 4 checks to evaluate the following: (1) Contamination of the first negative control (NDC); (2) Verification of the experimental positive control (CPC); (3) Verification of the internal positive control (RNaseP); (4) Contamination of the second negative control (NTC).
* ``Coinfection_Check_{IFC Barcode}.csv`` :  The description of the check for potential coinfection is explained in ``Quality_Control_Flags_{IFC Barcode}.csv``; however, this csv contains the tabular output of all samples that have been flagged as having potential coinfection, along with the viral assays that they may be coinfected with.
* ``Assay_Performance_QC_Tests_{IFC Barcode}.txt`` : Description of the Quality Control Tests done to evaluate assay performance in each experiment that is completed and analyzed. The first three use NTC (No Target Control) negativity, NDC (No Detection Control) negativity, and CPC (Combined Positive Control) positivity to assess each assay based on a scoring system from 0 to 1. The fourth test assesses the performance of the RNaseP (positive control used for human clinical samples) assay based on a scoring system from 0 to 1, generated by the positivity rate of clinical samples for the RNaseP assay.
* ``Assay_Level_QC_Metrics_{IFC Barcode}.csv`` : Tabular output of the scores of each assay for the Quality Control Tests done to evaluate assay performance in each experiment.

If you are more interested in research & development in the diagnostic space, additional files you might be interested are as follows:
* ``t{#}_{IFC Barcode}.csv`` : These are the quantitative signal output per timepoint. (For a 1-hour Standard Biotools Biomark run, you will generate 13 such total files corresponding to 13 timepoints.)
* ``Heatmap_t{#}_{IFC Barcode}.png`` : Heatmap for data visualization  per timepoint. (For a 1-hour Standard Biotools Biomark run, you will generate 13 such total files corresponding to 13 timepoints.)
* ``ref_norm.csv`` : This file corresponds to the signal of the passive reference ROX dye per timepoint, per assay used in your experiment.
* ``assigned_ref_norm.csv``: This is a more useful version of the ref_norm.csv as it shows you which sample and which assay the passive reference ROX dye corresponds to, at each timepoint.
* ``signal_norm.csv`` : This file corresponds to the normalized FAM signal (after subtraction of the background FAM signal and normalization against the signal from the reference dye).
* ``assigned_signal_norm.csv`` : This is a more useful version of the signal_norm.csv as it shows you which sample and which assay the normalized FAM signal corresponds to, at each timepoint.
* ``NTC_thresholds_{IFC Barcode}.csv`` : This file tabulates the mean NTC value thresholds above which samples were marked as positive.


## Congratulations on analyzing your CARMEN run! 
If you have any questions or run into any questions regarding the code base, please reach out to albeez@broadinstitute.org and cwilkason@broadinstitute.org. 

