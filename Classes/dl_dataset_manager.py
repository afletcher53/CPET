from dataclasses import dataclass
import os
import pandas as pd
from typing import Generic, TypeVar

from ProjectStrings import ProjectStrings

T = TypeVar('T')

@dataclass
class MortalityRecord:
    """
    A record contains breath by breath data, categorical data and the outcome
    """
    y: int
    x_bxb: pd.DataFrame
    x_cat: pd.DataFrame

class DLManager(Generic[T]):
    def __init__(self, record_type: type[T]) -> None:
        self.record_type = record_type
        self.strings = ProjectStrings()
        self.source = 'york'

    def load_data(self) -> list[T]:
        if self.record_type == MortalityRecord:
            return self.load_mortality_data()
 
    def load_mortality_data(self) -> list[MortalityRecord]:
        # Mortality data should have a specific format (Output is binary)
        
        # 1. get every file in the directory
        
        files = get_files(self.strings.york_binned_normalised, ".csv")

        # 2. Calculate the outcome for each file
        # Load up the outcome data
        

        # for each file, get the _single_variable_data.csv file from the upper directory
        mortality_data = []
        for file in files:
            research_id = os.path.basename(file).split("_")[0]
            mortality_data.append(
                MortalityRecord(

                    x_bxb=pd.read_csv(os.path.join(self.strings.york_binned_normalised, f"{research_id}_bxb.csv")),
                    x_cat=pd.read_csv(os.path.join(self.strings.york_dl, f"{research_id}_single_variable_data.csv")),
                    y = None
                )
            )


        return []

    def save_data(self, records: list[T]) -> None:
        # This is a placeholder implementation
        # You should replace this with actual data saving logic
        pass


def get_files(folder, extension):
    """Get all files in a folder with a given extension."""
    import os

    files = []
    for file in os.listdir(folder):
        if file.endswith(extension):
            files.append(os.path.join(folder, file))
    if not files:
        raise FileNotFoundError(
            f"No files with extension {extension} found in folder {folder}"
        )

    return files
# Example usage
mortality_manager = DLManager[MortalityRecord](MortalityRecord)
records = mortality_manager.load_data()