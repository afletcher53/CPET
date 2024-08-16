

import csv
import logging
import os

import pandas as pd
from Classes.ProjectStrings import ProjectStrings
from Classes.integrity_checks import Integrity_checks
import gather_sheffield_data
import gather_york_data
import init_project
import link_files
import anonymise

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
    # init_project.main()  # Initialise the project directories
    # check_recent_setup = init_project.LOCK_FILE

    # if check_recent_setup:  # Check if files have been placed, and if so, continue
    #     cpet_raw_empty = not os.path.exists(Strings.cpet_db) or (
    #         os.path.isdir(Strings.cpet_db) and len(os.listdir(Strings.cpet_db)) == 0)
    #     data_dir_empty = not os.path.exists(Strings.cpet_data) or (
    #         os.path.isdir(Strings.cpet_data) and len(os.listdir(Strings.cpet_data)) == 0)

    #     if cpet_raw_empty or data_dir_empty:
    #         if cpet_raw_empty:
    #             raise Exception(
    #                 "- Please add your SUM files to the cpet raw directory.")
    #         if data_dir_empty:
    #             raise Exception(
    #                 "- Please add your CPETdb.xlsx file to the data directory.")
    #     else:
    #         logger.info("Project initalised with correct file placements.")

    # Integrity_checks()


    # if not os.path.exists(Strings.linked_data_with_db):
    #     logger.info("Linked data file not found. Starting matching process.")
    #     link_files.main()
    # else:
    #     logger.info("Linked data file found. Using existing file.")

    # anonymise.main()

    # show missing BXB files

    # calculate how many of these files are "missing" bxb data, i.e. the file exists however there is less than 10 lines of data
    # missing_files = []
    # for file in files:
    #     file_name = file.split(".")[1]
    #     file_name = file_name.split("/")[-1]
    #     file_name = file_name + "_bxb_data.csv"

    #     data = open(os.path.join(ps.anonymised, file_name), "r").read()

    #     # Step 1: Remove leading '$' and strip whitespace
    #     cleaned_data = re.sub(r'^\$', '', data, flags=re.MULTILINE).strip()

    #     # Step 2: Split data into rows based on newlines
    #     rows = cleaned_data.split('\n')

    #     if len(rows) < 10:
    #         missing_files.append(file)

    # print(f"Number of missing files: {len(missing_files)}")
    # # order the missing files by name.sum which is numerical 
    # sorted_file_paths = sorted(missing_files, key=lambda x: int(x.split('/')[-1].split('.')[0]))

    # for file in sorted_file_paths:
    #     print(file)
    # print(f"Number of files to process: {len(files) - len(missing_files)}")
    # gather_york_data.main()
    gather_sheffield_data.main()

    

main()
