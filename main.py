

import csv
import logging
import os

import pandas as pd
from Classes.ProjectStrings import ProjectStrings
from Classes.integrity_checks import IntegrityChecks
from data_preparation import generate_mortality_data
import gather_york_data_dl
import gather_york_data_traditional
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

    # checks = IntegrityChecks()
    # checks.check_anon_sums(save=True)
    # logger.info("Starting data extraction.")
    # logger.info("Gathering York data for traditional analysis.")
    # gather_york_data_traditional.main()
    # logger.info("Gathering York data for DL analysis.")
    # gather_york_data_dl.main()
    # generate_mortality_data(days=365)
    # generate_mortality_data(days=180)
    # generate_mortality_data(days=90)
    generate_mortality_data(days=30)
main()
