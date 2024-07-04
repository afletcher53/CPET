import argparse
import csv
from datetime import datetime
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from Classes.mine_cpet_data import Raw_CPET_data
from Classes.ProjectStrings import ProjectStrings

PATIENT_ID = "PatientID"
BIRTHDAY = "Birthday"
VISIT_DATE_TIME = "VisitDateTime"
DATE_FORMAT = "%d/%m/%Y"


def parse_arguments():
    parser = argparse.ArgumentParser(description="CPET data processing script")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    return parser.parse_args()


def load_cpet_data(cpet_data_dir: str) -> List[Raw_CPET_data]:
    """Load CPET data from the given directory."""
    data = [Raw_CPET_data(file) for file in os.listdir(cpet_data_dir)]
    data = [d for d in data if d.file_name.endswith(".sum")]
    if not data:
        raise FileNotFoundError(f"No CPET files found in {cpet_data_dir}")
    return data


def format_date(date: datetime) -> str:
    """Format a datetime object to a string."""
    return date.strftime(DATE_FORMAT) if isinstance(date, datetime) else str(date)


def find_patient(
    cpet_data: List[Raw_CPET_data], patient_search_details: Dict[str, str]
) -> Tuple[Optional[str], str, List[Tuple[str, str, str]]]:
    """Find a patient in the CPET data based on search details."""
    identifiers = [
        ("patient ID_CPETdb", "patient ID CPETdb"),
        ("hospital number", "hospital number"),
        ("nhs number", "NHS number"),
        ("patient ID_CPETmachine", "patient ID CPETMachine"),
    ]

    all_matches = []
    operation_date = patient_search_details["Operation date"]

    filtered_cpet_data = []
    for data in cpet_data:
        is_within_6_months = data.is_within_6_months_of_operation(
            operation_date)
        if is_within_6_months:
            filtered_cpet_data.append(data)
        else:
            all_matches.append((data.file_name, "Discarded",
                               "Not within 6 months of operation"))

    cpet_data = filtered_cpet_data

    for id_key, match_detail in identifiers:
        for data in cpet_data:
            needle = data.read_column(PATIENT_ID)
            if needle.replace(" ", "") == str(patient_search_details[id_key]).replace(" ", "").replace("-", ""):
                all_matches.append((data.file_name, "Potential match",
                                   f"{match_detail}: {needle} == {patient_search_details[id_key]}, vdt: {data.read_column('VisitDateTimeAsDatetimeObject')}"))

    for data in cpet_data:
        if BIRTHDAY in data.columns and VISIT_DATE_TIME in data.columns:
            dob = data.read_column(BIRTHDAY)
            test_date = data.read_column(VISIT_DATE_TIME)

            search_dob = format_date(patient_search_details["Date of Birth"])
            search_test_date = format_date(patient_search_details["Test Date"])

            if dob == search_dob and test_date == search_test_date:
                all_matches.append((data.file_name, "Potential match",
                                   f"Date of birth ({dob}) and test date ({test_date}), vdt: {data.read_column('VisitDateTimeAsDatetimeObject')}"))

    if not all_matches:
        return None, "No match found", []

    def sort_key(match):
        filename, status, reason = match
        if status == "Discarded":
            return (1, reason)
        for idx, (id_key, _) in enumerate(identifiers):
            if id_key in reason:
                return (0, idx, patient_search_details["Test Date"])
        return (0, len(identifiers), patient_search_details["Test Date"])

    sorted_matches = sorted(all_matches, key=sort_key)

    best_match, status, best_reason = sorted_matches[0]
    if status == "Discarded":
        return None, "No valid match found", sorted_matches
    return best_match, f"Best match found: {best_reason}", sorted_matches


def write_mappings_to_csv(mappings: List[Dict], output_file: str):
    """Write patient mappings to a CSV file."""
    fieldnames = [
        "Research Number",
        "Patient ID",
        "Hospital Number",
        "NHS Number",
        "DOB",
        "Test Date",
        "CPET File",
        "Match Reason",
        "Discarded Matches",
        "Potential Matches"
    ]

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for mapping in mappings:
            discarded_matches = [f"{file}: {reason}" for file, status,
                                 reason in mapping["All Matches"] if status == "Discarded"]
            potential_matches = [f"{file}: {reason}" for file, status,
                                 reason in mapping["All Matches"] if status == "Potential match"]

            row = {
                "Research Number": mapping["Research Number"],
                "Patient ID": mapping["Patient ID"],
                "Hospital Number": mapping["Hospital Number"],
                "NHS Number": mapping["NHS Number"],
                "DOB": mapping["DOB"],
                "Test Date": mapping["Test Date"],
                "CPET File": mapping["CPET File"],
                "Match Reason": mapping["Match Reason"],
                "Discarded Matches": "; ".join(discarded_matches),
                "Potential Matches": "; ".join(potential_matches)
            }
            writer.writerow(row)


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
    strings = ProjectStrings()
    cpet_data = load_cpet_data(strings.cpet_data)

    cpet_data = [data for data in cpet_data if data.valid_bxb_section]

    cpet_db = pd.read_excel(strings.cpet_db)

    patient_search_details = cpet_db.to_dict(orient="records")

    mappings = []
    match_reasons_counter = Counter()

    patient_iterable = (
        tqdm(patient_search_details, desc="Processing patients")
        if use_tqdm
        else patient_search_details
    )

    for patient in patient_iterable:
        found_file, match_reason, all_matches = find_patient(
            cpet_data, patient)

        category = categorize_match_reason(match_reason)
        match_reasons_counter[category] += 1

        mappings.append(
            {
                "Research Number": int(patient["Research number"]),
                "Patient ID": patient["patient ID_CPETdb"],
                "Hospital Number": patient["hospital number"],
                "NHS Number": patient["nhs number"],
                "DOB": format_date(patient["Date of Birth"]),
                "Test Date": format_date(patient["Test Date"]),
                "CPET File": found_file if found_file else "Not Found",
                "Match Reason": match_reason,
                "All Matches": all_matches
            }
        )

    output_file = strings.linked_data
    write_mappings_to_csv(mappings, output_file)

    cpet_db_copy = cpet_db.copy()

    cpet_db_copy = cpet_db_copy.merge(
        pd.DataFrame(mappings),
        how="left",
        left_on="Research number",
        right_on="Research Number",
    )

    cpet_db_copy.to_csv(strings.linked_data_with_db, index=False)


if __name__ == "__main__":
    args = parse_arguments()
    main(use_tqdm=False)
