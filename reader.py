import pandas as pd
import io

class DataReader:
    def __init__(self):
        pass  # No initialization parameters needed as of now

    def clean_dataframe(self,df):
        # Drop 'Unnamed:' columns
        df = df.drop(columns=df.filter(like='Unnamed: ').columns)

        # Set 'Chamber ID' as index
        if 'Chamber ID' in df.columns:
            df = df.set_index('Chamber ID')

        # Strip leading spaces from column names
        df.columns = df.columns.str.lstrip()

        # Rename column names with 't' prefix
        df.columns = ['t' + str(col) for col in df.columns]
        
        return df

    def extract_dataframes_from_csv(self, file_like_object, phrases_to_find):
        # Placeholder for the indices
        indices = {}

        # Read the content of the file-like object (only once)
        file_like_object.seek(0)
        content = file_like_object.read().decode('utf-8')

        new_phrase = [
            "ref_raw", 
            "probe_raw",
            "ref_bkgd", 
            "probe_bkgd"
        ]
        phrase_mapping = dict(zip(phrases_to_find, new_phrase))

        # Split the content by lines and iterate through it to find the phrases
        lines = content.split('\n')
        for i, line in enumerate(lines):
            for phrase in phrases_to_find:
                if phrase in line:
                    indices[phrase_mapping[phrase]] = i + 1  # Add one to index so we eliminate the row with just the header

        # Calculate the number of rows in each section based on the first empty row
        section_rows = {}
        for phrase, start_index in indices.items():
            for i in range(start_index, len(lines)):
                if not lines[i].strip():  # Checks for an empty line
                    section_rows[phrase] = i - (start_index +1)
                    break
            else:
                section_rows[phrase] = len(lines) - (start_index)  # If no empty line is found, use all lines until the end

        # Read the specific sections from the CSV into dataframes
        dataframes = {}
        for phrase, start_index in indices.items():
            # Create a new file-like object from the content string for each read
            content_io = io.StringIO(content)
            df = pd.read_csv(content_io, nrows=section_rows[phrase], skiprows=start_index)
            dataframes[phrase] = self.clean_dataframe(df)  # Clean and store the dataframe
            #print(phrase, len(df))
        # Return a dictionary of the dataframes

        return dataframes