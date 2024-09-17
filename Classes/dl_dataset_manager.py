import os
import pandas as pd
from dataclasses import dataclass
from typing import Generic, TypeVar
import logging

from Classes.ProjectStrings import ProjectStrings

T = TypeVar("T")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("./logs/main.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

@dataclass
class MortalityRecord:
    y: int
    x_bxb: pd.DataFrame
    x_cat: pd.DataFrame

class DLManager(Generic[T]):
    def __init__(self, record_type: type[T], days: int = 365) -> None:
        self.record_type = record_type
        self.strings = ProjectStrings()
        self.source = "york"
        self.days = days
        self.output_folder = os.path.join("data", "ml_inputs", f"mortality_{days}")
        os.makedirs(self.output_folder, exist_ok=True)

    def load_data(self) -> list[T]:
        if self.record_type == MortalityRecord:
            return self.load_mortality_data()

    def load_mortality_data(self) -> list[MortalityRecord]:
        files = get_files(self.strings.york_binned_normalised, ".csv")
        outcomes = pd.read_excel(os.path.join(self.strings.york, "outcomes.xlsx"))

        outcomes["Date of death"] = pd.to_datetime(outcomes["Date of death"], errors="coerce")
        outcomes["dead_days"] = (outcomes["Date of death"] - outcomes["Date of operation"]).dt.days
        outcomes["dead_days"] = outcomes["dead_days"].fillna(outcomes["dead_days"].max())
        outcomes["y"] = outcomes["dead_days"].apply(lambda x: 1 if x < self.days else 0)

        logger.info(f"Total deaths within {self.days} days: {sum(outcomes['y'])}")

        mortality_data = []
        for file in files:
            research_id = int(os.path.basename(file).split("_")[0])
            id_column = "Research number" if "Research number" in outcomes.columns else "Research id"
            matching_rows = outcomes.loc[outcomes[id_column] == research_id]

            if matching_rows.empty:
                logger.info(f"Warning: No matching row found for research_id {research_id}")
                continue

            y_value = matching_rows["y"].values[0]

            mortality_data.append(
                MortalityRecord(
                    x_bxb=pd.read_csv(os.path.join(self.strings.york_dl, 'rpm_imputed', f"{research_id}_bxb.csv")),
                    x_cat=pd.read_csv(os.path.join(self.strings.york_dl, 'exercise_time_imputed', f"{research_id}_single_variable_data.csv")),
                    y=int(y_value),
                )
            )
        return mortality_data

def get_files(folder, extension):
    files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith(extension)]
    if not files:
        raise FileNotFoundError(f"No files with extension {extension} found in folder {folder}")
    return files