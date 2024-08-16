import os
import re
from Classes.ProjectStrings import ProjectStrings
import pandas as pd


def get_files(folder, extension):
    """Get all files in a folder with a given extension."""
    import os

    files = []
    for file in os.listdir(folder):
        if file.endswith(extension):
            files.append(os.path.join(folder, file))
    if not files:
        raise FileNotFoundError(
            f"No files with extension {extension} found in folder {folder}"
        )

    return files

def calculate_phase_of_breath(file, ps):
    """Calculate the phase of the breath for each row in the file."""
    
    file_name = file.split(".")[1]
    file_name = file_name.split("/")[-1]
    file_name = file_name + "_bxb_data.csv"

    data = open(os.path.join(ps.anonymised, file_name), "r").read()

    # Step 1: Remove leading '$' and strip whitespace
    cleaned_data = re.sub(r'^\$', '', data, flags=re.MULTILINE).strip()

    # Step 2: Split data into rows based on newlines
    rows = cleaned_data.split('\n')

    # Step 3: Split each row into columns based on commas
    data_list = [row.split(',') for row in rows]

    # Read the data into a DataFrame
    df = pd.DataFrame(data_list)

    # Drop all rows in column 0 that contain '"' (quotes)
    df = df[~df[0].str.contains('"')]

    # Drop column 0
    df = df.drop(columns=[0])

    # Drop row 0 (header row in this context)
    df = df.drop([0])

    # Rename specific columns
    df = df.rename(columns={2: "Breath Number", 9: "Load", 20: 'RPM'})

    # drop column 1
    df = df.drop(columns=[1])

    # drop the first two rows
    df = df.drop([2, 4])

    # conver all load and RPM values to numeric
    df['Load'] = pd.to_numeric(df['Load'])
    df['RPM'] = pd.to_numeric(df['RPM'])   

    # Calculate the phase of the breath. If RPM > 1 and Load = 0, pedalling with no load on bike, phase = 1, if RPM = 0 and load < 25 then phase = 0 (resting)
    # if RPM > 1 and Load > 1 start of test, phase = 2, if RPM = 0 and Load > 25 then phase = 3 (end of test)
    df['Phase'] = -1
    df.loc[(df['RPM'] == 0) & (df['Load'] < 25), 'Phase'] = 0
    df.loc[(df['RPM'] > 0) & (df['Load'] == 0), 'Phase'] = 1
    df.loc[(df['RPM'] > 0) & (df['Load'] > 0), 'Phase'] = 2



    try:
        # find the index of the peak load
        peak_load_index = df['Load'].idxmax()

        # for all phase 2 values after the peak load index, set the phase to 3
        df.loc[(df['Phase'] == 2) & (df.index > peak_load_index), 'Phase'] = 3
        
        # drop all columns except for breath number, load, RPM and phase
        df = df[['Breath Number', 'Load', 'RPM', 'Phase']]
        # duration of test is time from first phase 2 to first phase 3
        phase_2 = df[df['Phase'] == 2].index[0]
        phase_3 = df[df['Phase'] == 3].index[0]
        duration = phase_3 - phase_2
        print(f"Duration of test: {duration} breaths")

        # assert all of the phases are set
        assert -1 not in df['Phase'].values


        output_file = file_name.split(".")[0] + "_phase_of_breath.csv"
        output_file = os.path.join(ps.sheffield, output_file)
        df.to_csv(output_file, index=False)
    except:
    
        if df['RPM'].sum() == 0:
            print(f"RPM column is full of 0s: {file_name}")

        # check if phase 3 is missing
        if 3 not in df['Phase'].values:
            print(f"Phase 3 is missing: {file_name}")
        else:
            print(f"Error processing file: {file_name}")
    return df
def main():

    
    # for each sum file, lets calculate the phase of the breath
    ps = ProjectStrings()
    files = get_files(ps.anonymised, ".sum")

    # calculate how many of these files are "missing" bxb data, i.e. the file exists however there is less than 10 lines of data
    missing_files = []
    for file in files:
        file_name = file.split(".")[1]
        file_name = file_name.split("/")[-1]
        file_name = file_name + "_bxb_data.csv"

        data = open(os.path.join(ps.anonymised, file_name), "r").read()

        # Step 1: Remove leading '$' and strip whitespace
        cleaned_data = re.sub(r'^\$', '', data, flags=re.MULTILINE).strip()

        # Step 2: Split data into rows based on newlines
        rows = cleaned_data.split('\n')

        if len(rows) < 10:
            missing_files.append(file)

    print(f"Number of missing files: {len(missing_files)}")
    # order the missing files by name.sum which is numerical 
    sorted_file_paths = sorted(missing_files, key=lambda x: int(x.split('/')[-1].split('.')[0]))

    for file in sorted_file_paths:
        print(file)
    print(f"Number of files to process: {len(files) - len(missing_files)}")

    # remove the missing files from the list of files to process
    files = [file for file in files if file not in missing_files]

 
    for file in files:
        calculate_phase_of_breath(file, ps)





main()