import os
from .ProjectStrings import ProjectStrings


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
