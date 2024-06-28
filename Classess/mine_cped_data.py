import os

from ProjectStrings import ProjectStrings


class Raw_CPET_data:
    def __init__(self, file_name):
        strings = ProjectStrings()
        self.file_name = file_name  # Store the file name
        self.cpet_data_path = os.path.join(strings.cpet_data, file_name)
        self._check_file_exists()
        self.columns = self._get_columns()

    def _check_file_exists(self):
        if not os.path.exists(self.cpet_data_path):
            raise FileNotFoundError(f"File {self.cpet_data_path} not found")

    def _get_columns(self):
        res = self._parse_file(self.cpet_data_path)
        return res.keys()

    def _parse_file(self, file_path):
        result = {}
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split(';')
                if len(parts) >= 5:
                    key = parts[2]
                    value = parts[4]
                    result[key] = value
        return result

    def _check_column_exists(self, column_name):
        if column_name not in self.columns:
            raise ValueError(f"Column {column_name} not found")

    def read_column(self, column_name):
        self._check_column_exists(column_name)

        return self._read_column(column_name)

    def read_columns(self, column_names):
        self._check_columns_exist(column_names)
        return self._read_columns(column_names)
    
    def _read_column(self, column_name):
        res = self._parse_file(self.cpet_data_path)

        # if the columnName is VisitDateTime, return just the date
        if column_name == 'VisitDateTime':
            return res[column_name].split()[0]
        return res[column_name]
    
    def _read_columns(self, column_names):
        res = self._parse_file(self.cpet_data_path)
        # if the columnName is VisitDateTime, return just the date
        for key in res:
            if key == 'VisitDateTime':
                res[key] = res[key].split()[0]
                
        return {key: res[key] for key in column_names}
    
    def _check_columns_exist(self, column_names):
        for column_name in column_names:
            self._check_column_exists(column_name)

    def __str__(self):
        return f"Raw_CPET_data class with file {self.cpet_data_path}"
import os
from typing import Dict, List, Optional, Tuple

from ProjectStrings import ProjectStrings


def load_cpet_data(cpet_data_dir: str) -> List[Raw_CPET_data]:
    return [Raw_CPET_data(file) for file in os.listdir(cpet_data_dir)]

def find_patient(cpet_data: List[Raw_CPET_data], patient_search_details: Dict[str, str]) -> Tuple[Optional[str], str]:
    for data in cpet_data:
        patient_id = data.read_column('PatientID')
        if patient_id in [patient_search_details['id'], 
                          patient_search_details['hospital_num'], 
                          patient_search_details['nhs_num']]:
            match_type = 'ID'
            
            if patient_id == patient_search_details['id']:
                match_detail = 'id'
            elif patient_id == patient_search_details['hospital_num']:
                match_detail = 'hospital number'
            else:
                match_detail = 'NHS number'
            return data.file_name, f"Matched by {match_detail}"
        
        # check that the Birthday Column exists 
        if 'Birthday' in data.columns or 'VisitDateTime' in data.columns:
            print(data.read_column('Birthday'))
            print(data.read_column('VisitDateTime'))
            print(patient_search_details['date_of_birth'])
            print(patient_search_details['date_of_test'])
            if (data.read_column('Birthday') == patient_search_details['date_of_birth'] and
                data.read_column('VisitDateTime') == patient_search_details['date_of_test']):
                print(f"Found patient with matching DOB and test date in file {data.file_name}")

                # Check for other patients with the same DOB and test date
                for data2 in cpet_data:
                    if (data2 != data and
                        data2.read_column('Birthday') == patient_search_details['date_of_birth'] and
                        data2.read_column('VisitDateTime') == patient_search_details['date_of_test']):
                        print(f"Another patient with matching DOB and test date in file {data2.file_name}")
                        return None, "Multiple matches found with the same DOB and test date"

                return data.file_name, "Matched by date of birth and test date"

    return None, "No match found"

def main():
    strings = ProjectStrings()
    cpet_data = load_cpet_data(strings.cpet_data)

    patient_search_details = {
        'id': '',
        'hospital_num': 'df5678',
        'nhs_num': '4794533123',   
        'date_of_birth': '08/08/1976',
        'date_of_test': '25/09/2008',
    }

    found_file, match_reason = find_patient(cpet_data, patient_search_details)

    if found_file:
        print(f"The patient's data was found in the file: {found_file}")
        print(f"Match reason: {match_reason}")
    else:
        print(f"No matching file found for the patient. {match_reason}")

if __name__ == "__main__":
    main()