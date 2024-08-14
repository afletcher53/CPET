

import csv
import logging
import os

import pandas as pd
from Classes.ProjectStrings import ProjectStrings
from Classes.integrity_checks import Integrity_checks
import gather_sheffield_data
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
    gather_sheffield_data.main()

main()
