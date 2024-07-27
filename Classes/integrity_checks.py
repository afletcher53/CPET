import logging
import os
from Classes.ProjectStrings import ProjectStrings
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("./logs/main.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)


class Integrity_checks:

    def __init__(self):
        self.strings = ProjectStrings()
        self.cpet_raw_files = self._find_raw_files()
        self._check_raw_files_extensions()
        self._check_cpet_db()

    def _find_raw_files(self):
        """
        Find the raw files within the data cpet raw directory
        """
        sum_files_dir = self.strings.cpet_data
        cpet_raw_files = [file for file in os.listdir(sum_files_dir)]
        return cpet_raw_files

    def _check_raw_files_extensions(self):
        """
        Check the extensions of the raw files are all .sum
        """
        for file in self.cpet_raw_files:
            if not file.endswith(".sum"):
                raise Exception(
                    f"File {file} does not have the correct extension. Please ensure all files are .sum")
        logger.info("All CPET files have the correct extension")

    def _check_cpet_db(self):
        """
        Check the CPETdb.xlsx file is in the correct directory
        """
        if not os.path.exists(self.strings.cpet_db):
            raise Exception(
                f"CPETdb.xlsx file not found in {self.strings.cpet_db}")
        logger.info("CPETdb.xlsx file found")

        # check that all the required columns are present

        cpet_db = pd.read_excel(self.strings.cpet_db)
        required_columns = self.strings.required_cpet_db_columns
        for column in required_columns:
            if column not in cpet_db.columns:
                raise Exception(
                    f"Column {column} not found in CPETdb.xlsx file")

        logger.info("All required columns found in CPETdb.xlsx file")

        # check that the required columns are not NaN
        required_columns_not_nan = self.strings.required_cpet_data_not_nan_columns
        for column in required_columns_not_nan:
            if cpet_db[column].isnull().values.any():
                research_number = cpet_db[cpet_db[column].isnull(
                )]["Research number"]
                logger.error(
                    f"Column {column} has NaN values in CPETdb.xlsx file for research number(s): {research_number}")
                raise Exception(
                    f"Column {column} has NaN values in CPETdb.xlsx file for research number(s): {research_number}")
        logger.info(
            "All required columns in CPETdb.xlsx file have no NaN values")
