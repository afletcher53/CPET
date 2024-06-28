from collections import Counter
import csv
import os
import logging
from logging.handlers import RotatingFileHandler
from typing import List, Dict, Tuple, Optional
import datetime
import pandas as pd
from functools import partial

import argparse
from tqdm import tqdm

from Classes.ProjectStrings import ProjectStrings
from Classes.mine_cped_data import Raw_CPET_data

# Constants
PATIENT_ID = 'PatientID'
BIRTHDAY = 'Birthday'
VISIT_DATE_TIME = 'VisitDateTime'
DATE_FORMAT = '%d/%m/%Y'

# Configure logging


def setup_logging(log_level):
    # Create logs directory if it doesn't exist
    if not os.path.exists('./logs'):
        os.makedirs('./logs')

    # Set up root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File handler (with rotation)
    file_handler = RotatingFileHandler(
        './logs/cpet_processing.log', maxBytes=10*1024*1024, backupCount=5)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def parse_arguments():
    parser = argparse.ArgumentParser(description="CPET data processing script")
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help="Set the logging level")
    return parser.parse_args()


def lazy_log(level, msg, *args, **kwargs):
    if logger.isEnabledFor(level):
        logger.log(level, msg, *args, **kwargs)


lazy_info = partial(lazy_log, logging.INFO)
lazy_error = partial(lazy_log, logging.ERROR)


def load_cpet_data(cpet_data_dir: str) -> List[Raw_CPET_data]:
    """Load CPET data from the given directory."""
    try:
        data = [Raw_CPET_data(file) for file in os.listdir(cpet_data_dir)]
        # only return .sum files
        data = [d for d in data if d.file_name.endswith('.sum')]

        if not data:
            raise FileNotFoundError(f"No CPET files found in {cpet_data_dir}")
        return data
    except Exception as e:
        lazy_error("Error loading CPET data: %s", e)
        raise


def format_date(date: datetime.datetime) -> str:
    """Format a datetime object to a string."""
    return date.strftime(DATE_FORMAT) if isinstance(date, datetime.datetime) else str(date)


def find_patient(cpet_data: List[Raw_CPET_data], patient_search_details: Dict[str, str]) -> Tuple[Optional[str], str]:
    """Find a patient in the CPET data based on search details."""
    identifiers = [
        ('patient ID_CPETdb', 'patient ID CPETdb'),
        ('hospital number', 'hospital number'),
        ('nhs number', 'NHS number'),
        ('patient ID_CPETmachine', 'patient ID CPETMachine')
    ]

    matches = []
    match_reasons = []

    for id_key, match_detail in identifiers:
        for data in cpet_data:
            needle = data.read_column(PATIENT_ID)
            if needle.replace(" ", "") == str(patient_search_details[id_key]).replace(" ", "").replace("-", ""):
                matches.append(data.file_name)
                match_reasons.append(
                    f"{match_detail}: {needle} == {patient_search_details[id_key]}")

    # Check for DOB and test date match
    for data in cpet_data:
        if BIRTHDAY in data.columns and VISIT_DATE_TIME in data.columns:
            dob = data.read_column(BIRTHDAY)
            test_date = data.read_column(VISIT_DATE_TIME)

            search_dob = format_date(patient_search_details['Date of Birth'])
            search_test_date = format_date(patient_search_details['Test Date'])

            if dob == search_dob and test_date == search_test_date:
                matches.append(data.file_name)
                match_reasons.append(
                    f"Date of birth ({dob}) and test date ({test_date})")

    if not matches:
        return None, "No match found"

    # Sort matches by priority in identifiers and then date_of_test
    def sort_key(match):
        for idx, (id_key, _) in enumerate(identifiers):
            if any(reason.startswith(id_key) for reason in match_reasons):
                return (idx, patient_search_details['Test Date'])
        return (len(identifiers), patient_search_details['Test Date'])

    sorted_matches = sorted(zip(matches, match_reasons),
                            key=lambda x: sort_key(x[1]))

    # Log the report of the matches
    lazy_info("\n" + "-" * 50)
    lazy_info("Patient ID = %s", patient_search_details['patient ID_CPETdb'])
    lazy_info("Hospital Number = %s",
              patient_search_details['hospital number'])
    lazy_info("NHS Number = %s", patient_search_details['nhs number'])
    lazy_info("DOB = %s", patient_search_details['Date of Birth'])
    lazy_info("Test Date = %s", patient_search_details['Test Date'])
    lazy_info("Matches found:")
    for match, reason in sorted_matches:
        lazy_info("%s: %s", match, reason)

    # Return the best match and the reason
    best_match, best_reason = sorted_matches[0]
    return best_match, f"Best match found: {best_reason}"


def write_mappings_to_csv(mappings: List[Dict], output_file: str):
    """Write patient mappings to a CSV file."""
    fieldnames = ['Patient ID', 'Hospital Number', 'NHS Number',
                  'Research Number', 'DOB', 'Test Date', 'CPET File', 'Match Reason']

    try:
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for mapping in mappings:
                writer.writerow(mapping)
        lazy_info("Mappings successfully written to %s", output_file)
    except Exception as e:
        lazy_error("Error writing mappings to CSV: %s", e)


def categorize_match_reason(reason: str) -> str:
    if "patient ID CPETdb" in reason:
        return "Patient ID CPETdb match"
    elif "hospital number" in reason:
        return "Hospital number match"
    elif "NHS number" in reason:
        return "NHS number match"
    elif "patient ID CPETMachine" in reason:
        return "Patient ID CPETMachine match"
    elif "Date of birth" in reason and "test date" in reason:
        return "Date of birth and test date match"
    elif "No match found" in reason:
        return "No match found"
    else:
        return "Other match"


def main(use_tqdm=False):
    """Main function to process CPET data and find patient matches."""
    if use_tqdm:
        lazy_info("Using tqdm for progress bar")
        logger.disabled = True
    try:
        lazy_info("Starting CPET data processing")
        strings = ProjectStrings()
        cpet_data = load_cpet_data(strings.cpet_data)
        cpet_db = pd.read_excel(strings.cpet_db)

        patient_search_details = cpet_db.to_dict(orient='records')

        mappings = []
        match_reasons_counter = Counter()

        # Create an iterable (either tqdm or the original list)
        patient_iterable = tqdm(
            patient_search_details, desc="Processing patients") if use_tqdm else patient_search_details

        for patient in patient_iterable:
            if not use_tqdm:
                lazy_info("\n" + "-" * 50)
                lazy_info("Searching for patient with details:")
                lazy_info("Patient ID = %s", patient['patient ID_CPETdb'])
                lazy_info("Hospital Number = %s", patient['hospital number'])
                lazy_info("NHS Number = %s", patient['nhs number'])
                lazy_info("Research Number = %s", patient['Research number'])
                lazy_info("DOB = %s", patient['Date of Birth'])
                lazy_info("Test Date = %s", patient['Test Date'])

            found_file, match_reason = find_patient(cpet_data, patient)

            if not use_tqdm:
                if found_file:
                    lazy_info(
                        "\nThe patient's data was found in the file: %s", found_file)
                    lazy_info("Match reason: %s", match_reason)
                else:
                    lazy_info(
                        "\nNo matching file found for the patient. %s", match_reason)

            category = categorize_match_reason(match_reason)
            match_reasons_counter[category] += 1

            mappings.append({
                'Research Number': int(patient['Research number']),
                'Patient ID': patient['patient ID_CPETdb'],
                'Hospital Number': patient['hospital number'],
                'NHS Number': patient['nhs number'],
                'DOB': format_date(patient['Date of Birth']),
                'Test Date': format_date(patient['Test Date']),
                'CPET File': found_file if found_file else 'Not Found',
                'Match Reason': match_reason
            })

        # Write mappings to CSV
        output_file = strings.linked_data
        write_mappings_to_csv(mappings, output_file)

        # Log the count of each category of match
        lazy_info("\nMatch categories and their counts:")
        for category, count in match_reasons_counter.items():
            lazy_info("%s: %d", category, count)

        # Log total number of patients processed
        lazy_info("\nTotal number of patients processed: %d", len(mappings))
        lazy_info("CPET data processing completed")

    except Exception as e:
        lazy_error("An error occurred in the main function: %s", e)


if __name__ == "__main__":
    args = parse_arguments()
    if args.log_level:
        log_level = getattr(logging, args.log_level)
        logger = setup_logging(log_level)
        main(use_tqdm=False)
    else:
        logger = setup_logging(logging.INFO)
        main(use_tqdm=True)
