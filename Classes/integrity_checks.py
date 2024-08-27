import logging
import os

from Classes.ProjectStrings import ProjectStrings
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple

import numpy as np


class IntegrityChecks:
    def __init__(self):
        self.strings = ProjectStrings()
        self.logger = self._setup_logger()
        self.cpet_raw_files = self._find_raw_files()
        self._run_initial_checks()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        file_handler = logging.FileHandler(os.path.join(self.strings.logs, "main.log"))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    def _find_raw_files(self) -> List[str]:
        return [
            file for file in os.listdir(self.strings.cpet_data) if file.endswith(".sum")
        ]

    def _run_initial_checks(self):
        self._check_raw_files_extensions()
        self._check_cpet_db()
        self.logger.info("All initial integrity checks passed")

    def _check_raw_files_extensions(self):
        invalid_files = [
            file for file in self.cpet_raw_files if not file.endswith(".sum")
        ]
        if invalid_files:
            raise ValueError(
                f"Files with incorrect extensions: {', '.join(invalid_files)}"
            )
        self.logger.info("All CPET files have the correct extension")

    def _check_cpet_db(self):
        if not os.path.exists(self.strings.cpet_db):
            raise FileNotFoundError(
                f"CPETdb.xlsx file not found in {self.strings.cpet_db}"
            )
        self.logger.info("CPETdb.xlsx file found")

        cpet_db = pd.read_excel(self.strings.cpet_db)
        missing_columns = set(self.strings.required_cpet_db_columns) - set(
            cpet_db.columns
        )
        if missing_columns:
            raise ValueError(
                f"Missing columns in CPETdb.xlsx: {', '.join(missing_columns)}"
            )
        self.logger.info("All required columns found in CPETdb.xlsx file")

        for column in self.strings.required_cpet_data_not_nan_columns:
            nan_research_numbers = cpet_db.loc[
                cpet_db[column].isnull(), "Research number"
            ]
            if not nan_research_numbers.empty:
                raise ValueError(
                    f"Column {column} has NaN values for research number(s): {', '.join(map(str, nan_research_numbers))}"
                )
        self.logger.info("All required columns in CPETdb.xlsx file have no NaN values")

    def check_anon_sums(
        self, save: bool = False, halt_on_errors: bool = False
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        files = sorted(
            [
                file
                for file in os.listdir(self.strings.anonymised)
                if file.endswith(".sum")
            ],
            key=lambda x: int(x.split(".")[0]),
        )

        missing_files = []
        missing_phase_3 = []
        missing_rpm = []
        missing_no_reasons = []

        for file in tqdm(
            files, desc="Checking Integrities of anonymised files", unit="file"
        ):
            missing_file = self._check_valid_bxb_section_in_anon(file)
            if missing_file:
                missing_files.append(missing_file)
                continue
            self._analyse_phase_of_breath(
                file, missing_phase_3, missing_rpm, missing_no_reasons
            )

        self._log_missing_files(
            missing_files, missing_phase_3, missing_rpm, missing_no_reasons
        )

        if save:
            self._save_missing_files(
                missing_files, missing_phase_3, missing_rpm, missing_no_reasons
            )

        if halt_on_errors and any(
            [missing_files, missing_phase_3, missing_rpm, missing_no_reasons]
        ):
            raise ValueError("Integrity checks failed")

        return missing_files, missing_phase_3, missing_rpm, missing_no_reasons

    def _check_valid_bxb_section_in_anon(self, file: str) -> str:
        file_name = f"{file.split('.')[0]}_bxb_data.csv"
        file_path = os.path.join(self.strings.anonymised, file_name)

        try:
            with open(file_path, "r") as f:
                data = f.read()
            cleaned_data = data.replace("$", "").strip()
            rows = cleaned_data.split("\n")
            return file_name if len(rows) < 10 else None
        except FileNotFoundError:
            self.logger.error(f"File not found: {file_path}")
            return file_name

    def _analyse_phase_of_breath(
        self,
        file: str,
        missing_phase_3: List[str],
        missing_rpm: List[str],
        missing_no_reasons: List[str],
    ):
        file_name = f"{file.split('.')[0]}_bxb_data.csv"
        file_path = os.path.join(self.strings.anonymised, file_name)

        try:
            df = pd.read_csv(file_path, header=None, skiprows=1)

            if len(df.columns) < 21:
                raise ValueError(f"Unexpected number of columns in {file_name}")

            df = df.rename(columns={2: "Breath Number", 9: "Load", 20: "RPM"})
            df = df[["Breath Number", "Load", "RPM"]]

            df["Load"] = df["Load"].replace("Watts", np.nan)
            df["RPM"] = df["RPM"].replace("RPM", np.nan)

            df = df.apply(pd.to_numeric, errors="coerce")

            df = df.dropna()

            df = df.reset_index(drop=True)
            # phase 0 = No RPM, No Load
            # phase 1 = RPM, No Load
            # phase 2 = RPM, Load
            # phase 3 = RPM, Load after peak load

            df = df.drop(0) # drop the first row as it is not a breath
            df["Phase"] = 0
            df.loc[(df["RPM"] > 0) & (df["Load"] == 0), "Phase"] = 1
            df.loc[(df["RPM"] > 0) & (df["Load"] > 0), "Phase"] = 2
            peak_load_index = df["Load"].idxmax()
            df.loc[(df["Phase"] == 2) & (df.index > peak_load_index), "Phase"] = 3
            


            if df["RPM"].sum() == 0:
                missing_rpm.append(file_name)
            elif 3 not in df["Phase"].values:
                missing_phase_3.append(file_name)
            elif -1 in df["Phase"].values:
                missing_no_reasons.append(file_name)
            else:
                output_file = os.path.join(
                    self.strings.sheffield, f"{file.split('.')[0]}_phase_of_breath.csv"
                )
                df.to_csv(output_file, index=False)
        except Exception as e:
            missing_no_reasons.append(file_name)

    def _log_missing_files(
        self,
        missing_files: List[str],
        missing_phase_3: List[str],
        missing_rpm: List[str],
        missing_no_reasons: List[str],
    ):
        self.logger.info(f"Number of missing files: {len(missing_files)}")
        self.logger.info(f"Number of missing phase 3: {len(missing_phase_3)}")
        self.logger.info(f"Number of missing RPM: {len(missing_rpm)}")
        self.logger.info(
            f"Number of missing files with no reason: {len(missing_no_reasons)}"
        )

    def _save_missing_files(
        self,
        missing_files: List[str],
        missing_phase_3: List[str],
        missing_rpm: List[str],
        missing_no_reasons: List[str],
    ):
        self._write_list_to_file(self.strings.missing_bxb, missing_files)
        self._write_list_to_file(self.strings.missing_phase_3, missing_phase_3)
        self._write_list_to_file(self.strings.missing_rpm, missing_rpm)
        self._write_list_to_file(self.strings.missing_no_reason, missing_no_reasons)

    def _write_list_to_file(self, file_path: str, items: List[str]):
        with open(file_path, "w") as f:
            for item in items:
                f.write(f"{item}\n")
        self.logger.info(f"{len(items)} items written to {file_path}")
