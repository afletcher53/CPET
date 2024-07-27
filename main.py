

import csv
import logging
import os

import pandas as pd
from Classes.ProjectStrings import ProjectStrings
from Classes.integrity_checks import Integrity_checks
import init_project
import link_files
import shutil

Strings = ProjectStrings()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("./logs/main.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)


def main():
    init_project.main()  # Initialise the project directories
    check_recent_setup = init_project.LOCK_FILE

    if check_recent_setup:  # Check if files have been placed, and if so, continue
        cpet_raw_empty = not os.path.exists(Strings.cpet_db) or (
            os.path.isdir(Strings.cpet_db) and len(os.listdir(Strings.cpet_db)) == 0)
        data_dir_empty = not os.path.exists(Strings.cpet_data) or (
            os.path.isdir(Strings.cpet_data) and len(os.listdir(Strings.cpet_data)) == 0)

        if cpet_raw_empty or data_dir_empty:
            if cpet_raw_empty:
                raise Exception(
                    "- Please add your SUM files to the cpet raw directory.")
            if data_dir_empty:
                raise Exception(
                    "- Please add your CPETdb.xlsx file to the data directory.")
        else:
            logger.info("Project initalised with correct file placements.")

    Integrity_checks()

    logger.info("All integrity checks passed")

    if not os.path.exists(Strings.linked_data_with_db):
        logger.info("Linked data file not found. Starting matching process.")
        link_files.main()
    else:
        logger.info("Linked data file found. Using existing file.")

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
    for index, row in linked_data_with_db_copy.iterrows():
        research_number = row['Research number']
        sum_file = f"{research_number}.sum"
        sum_file_path = os.path.join(Strings.anonymised, sum_file)
        with open(sum_file_path, 'r') as file:
            age = None
            bmi = None
            heart_rates = None

            lines = file.readlines()
            for line in lines:
                if '$;6025;Age;' in line:  # Look for the specific Age field
                    age = line.split(';')[-1].strip()
                    linked_data_with_db_copy.loc[linked_data_with_db_copy['Research number']
                                                 == research_number, 'Sum_Age'] = age
                if '$;6100;BMI;' in line:  # Look for the specific BMI field
                    bmi = line.split(';')[-1].strip()
                    linked_data_with_db_copy.loc[linked_data_with_db_copy['Research number']
                                                 == research_number, 'Sum_BMI'] = bmi
                if '$;3068;HR BPM' in line:
                    parts = line.split(';')
                    start_index = parts.index('HR BPM') + 1
                    heart_rates = [
                        part for part in parts[start_index:] if part.strip()]
                    heart_rates_str = ','.join(heart_rates)
                    # remove any \n characters
                    heart_rates_str = heart_rates_str.replace('\n', '')

                    linked_data_with_db_copy.loc[linked_data_with_db_copy['Research number']
                                                 == research_number, 'Sum_HR_BPM'] = heart_rates_str

                if '$;3903;Chronotropic Index' in line:
                    chronotropic_index = line.split(';')[-1].strip()
                    linked_data_with_db_copy.loc[linked_data_with_db_copy['Research number']
                                                 == research_number, 'Sum_Chronotropic_Index'] = chronotropic_index
        # extract the BXB section and add it to a pandas dataframe, they start from $;4000;BxBSection;; and end $;4999;BxBSectionEnd;;
        bxb_section = False
        bxb_data = []
        for line in lines:
            if '$;4000;BxBSection;;' in line:
                bxb_section = True
            if '$;4999;BxBSectionEnd;;' in line:
                bxb_section = False
            if bxb_section:
                bxb_data.append(line)
        bxb_data = [line.split(';') for line in bxb_data]
        # in line 3, pop the 3rd element, to align the data with the field names
        bxb_data[1].pop(2)

        # save the bxb data to a csv file with the research number as the name
        filename = f"./data/anonymised/{research_number}_bxb_data.csv"
        with open(filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerows(bxb_data)

    linked_data_with_db_copy.to_csv(
        Strings.anonymised_linked_data_with_db, index=False)


main()
