from datetime import datetime
import os
from .ProjectStrings import ProjectStrings


class Raw_CPET_data:
    def __init__(self, file_name):
        strings = ProjectStrings()
        self.file_name = file_name  # Store the file name
        self.cpet_data_path = os.path.join(strings.cpet_data, file_name)
        self._check_file_exists()
        self.columns = self._get_columns()
        self.valid_bxb_section = self._check_bxb_section()

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
        if column_name not in self.columns and column_name != 'VisitDateTimeAsDatetimeObject':
            raise ValueError(f"Column {column_name} not found")

    def read_column(self, column_name):
        self._check_column_exists(column_name)

        return self._read_column(column_name)

    def read_columns(self, column_names):
        self._check_columns_exist(column_names)
        return self._read_columns(column_names)

    def _read_column(self, column_name):
        res = self._parse_file(self.cpet_data_path)

        if column_name == 'VisitDateTime':
            return res[column_name].split()[0]

        if column_name == 'VisitDateTimeAsDatetimeObject':
            return datetime.strptime(res['VisitDateTime'], '%d/%m/%Y %H:%M:%S')
        return res[column_name]

    def _read_columns(self, column_names):
        res = self._parse_file(self.cpet_data_path)

        for key in res:
            if key == 'VisitDateTime':
                res[key] = res[key].split()[0]

        return {key: res[key] for key in column_names}

    def _check_columns_exist(self, column_names):
        for column_name in column_names:
            self._check_column_exists(column_name)

    def _check_bxb_section(self):
        with open(self.cpet_data_path, 'r') as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                if line.strip() == '$;4000;BxBSection;;':
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        parts = next_line.split(';')
                        if len(parts) >= 4:
                            values = parts[3].split(';')
                            return all(value.isdigit() for value in values)
        return False

    def is_within_6_months_of_operation(self, operation_date):
        visit_date = self.read_column('VisitDateTime')
        return self._is_within_6_months(visit_date, operation_date)

    def _is_within_6_months(self, visit_date, operation_date):
        if isinstance(visit_date, str):
            visit_date = datetime.strptime(visit_date, '%d/%m/%Y')
        if isinstance(operation_date, str):
            operation_date = datetime.strptime(operation_date, '%d/%m/%Y')

        difference = abs((operation_date - visit_date).days)
        return difference <= 180

    def __str__(self):
        return f"Raw_CPET_data class with file {self.cpet_data_path}"
