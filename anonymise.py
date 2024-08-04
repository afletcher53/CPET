
import csv
import logging
import os
import re
import shutil
import pandas as pd

from Classes.ProjectStrings import ProjectStrings
Strings = ProjectStrings()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("./logs/main.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

def extract_value(content, identifier):
    part = content.split(identifier)[1].split('$;')[0]
    return part.split(';')[-1].strip()

def extract_gxt_features(content, identifier):
    gxt_start = content.find('$;1000;GXTestDataSection;')
    gxt_end = content.find('$;3999;GXTestDataSectionEnd;;')
    if gxt_start == -1 or gxt_end == -1:
        print("Warning: GXT section markers not found in the content.")
        return {}
    content = content[gxt_start:gxt_end]
    lines = content.strip().split('\n')
    header = ['Combined'] + lines[0].split(';')[3:]
    data = []
    for line in lines[1:]:
        parts = line.split(';')
        combined = ';'.join(parts[:3]) + ';' # Concatenate first 3 columns
        values = [combined] + parts[3:]
        data.append(values)
    df = pd.DataFrame(data, columns=header)
    for column in df.columns[1:]:
        df[column] = pd.to_numeric(df[column], errors='ignore')
    df.set_index('Combined', inplace=True)
    row = df.loc[identifier]
    del df
    row_dict = row.to_dict()
    row_list = [(key, value) for key, value in row_dict.items()]
    return row_list



def process_file(file_path, features_map_to_extract):
    with open(file_path, 'r') as file:
        content = file.read()
    
    data = {}
    for feature, identifier in features_map_to_extract.items():
        if identifier in content:
            if feature in Strings.gxt_features:
                values = extract_gxt_features(content, identifier)
                for column, value in values:
                    data[f"{feature} {column}"] = value
            else:
                data[feature] = extract_value(content, identifier)
    
    # Extract BxB data as a single csv
    bxb_start = content.find('$;4000;BxBSection;;')
    bxb_end = content.find('$;4999;BxBSectionEnd;;')
    if bxb_start != -1 and bxb_end != -1:
        bxb_data = content[bxb_start:bxb_end].split('\n')
        bxb_data = [line.split(';') for line in bxb_data if line]
        if len(bxb_data) > 1:
            bxb_data[1].pop(2)
    else:
        bxb_data = []
    
    return data, bxb_data

def process_data(linked_data_with_db_copy, features_map_to_extract):
    results = []
    bxb_data_all = []

    for _, row in linked_data_with_db_copy.iterrows():
        research_number = row['Research number']
        sum_file = f"{research_number}.sum"
        sum_file_path = os.path.join(Strings.anonymised, sum_file)
        
        data, bxb_data = process_file(sum_file_path, features_map_to_extract)
        data['Research number'] = research_number
        results.append(data)
        
        if bxb_data:
            bxb_data_all.append((research_number, bxb_data))
    
    # Update the dataframe
    results_df = pd.DataFrame(results)
    results_df.columns = ['SUM_' + col if col != 'Research number' else col for col in results_df.columns]

    linked_data_with_db_copy = linked_data_with_db_copy.merge(results_df, on='Research number', how='left')
    

    # column_mapping = {column: f"SUM_{column}" for column in results_df.columns if column != 'Research number'}
    # linked_data_with_db_copy.rename(columns=column_mapping, inplace=True)
    
    # Write BxB data
    for research_number, bxb_data in bxb_data_all:
        filename = f"./data/anonymised/{research_number}_bxb_data.csv"
        pd.DataFrame(bxb_data).to_csv(filename, index=False, header=False)
    
    return linked_data_with_db_copy
def main():
    
    # create a new DF with each matched row
    df = pd.read_csv(Strings.linked_data_with_db)
    matched_df = df[df['CPET File'] != 'Not Found']

    logger.info("Matched data:")
    logger.info(matched_df)

    # for each matched row, we want to move the corresponding matched SUM file to the anonymised directory and rename it to the research number

    linked_data_with_db_copy = pd.read_csv(Strings.linked_data_with_db)
    # drop linked data with db copy columns Date of Birth, hospital number, case note number 1, case note number 2, patient ID_CPETdb patient ID_CPETdbmachine
    linked_data_with_db_copy.drop(
        columns=['Date of Birth', 'hospital number', 'nhs number', 'case note number 1', 'case note number 2', 'patient ID_CPETdb', 'patient ID_CPETmachine', 'Patient ID', 'Hospital Number', 'NHS Number', 'DOB', 'Test Date_y', 'CPET File', 'Match Reason', 'All Matches'
                 ], inplace=True)

    for index, row in matched_df.iterrows():
        research_number = row['Research number']
        cpet_file = row['CPET File']
        cpet_file_path = os.path.join(Strings.cpet_data, cpet_file)
        anonymised_file_path = os.path.join(
            Strings.anonymised, f"{research_number}.sum")

        shutil.copy2(cpet_file_path, anonymised_file_path)

        patient_info_fields = ['PatientID', 'PatientLastName',
                               'PatientFirstName', 'PatientMiddleName', 'PatientFullName', 'Birthday']

        with open(anonymised_file_path, 'r') as file:
            lines = file.readlines()

        with open(anonymised_file_path, 'w') as file:
            for line in lines:
                if not any(field in line for field in patient_info_fields):
                    file.write(line)
        linked_data_with_db_copy.loc[linked_data_with_db_copy['Research number']
                                     == research_number, 'CPET File'] = f"{research_number}.sum"

        # copy accross any BXB data from the sum file to the linked data with db copy

        logger.info(
            f"Copied and anonymized {cpet_file} to {research_number}.sum")

    # drop any rows that have not been matched
    linked_data_with_db_copy = linked_data_with_db_copy.dropna(subset=[
                                                               'CPET File'])

    linked_data_with_db_copy.to_csv(
        Strings.anonymised_linked_data_with_db, index=False)

    # copy across heart rate data from the sum files to the linked data with db copy
    linked_data_with_db_copy = pd.read_csv(
        Strings.anonymised_linked_data_with_db)
    linked_data_with_db_copy['Heart Rate'] = None
    with open(Strings.sum_features, 'r') as file:
        features_map = file.readlines()
    features_map = [feature.strip() for feature in features_map]
    features_map_to_extract = Strings.wanted_feature_maps(features = features_map)
    linked_data_with_db_copy = process_data(linked_data_with_db_copy, features_map_to_extract)
    linked_data_with_db_copy.to_csv(Strings.anonymised_linked_data_with_db, index=False)


if __name__ == "__main__":
    main()
