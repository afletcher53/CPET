import os
import re


class ProjectStrings:
    _shared_state = {}

    def __init__(self):
        self.__dict__ = self._shared_state
        if not self._shared_state:
            self.data_path = "./data/"
            self.cpet_data = os.path.join(self.data_path, 'cpet raw')
            self.anonymised = os.path.join(self.data_path, 'anonymised')
            self.york = os.path.join(self.anonymised, 'york')
            self.york_traditional = os.path.join(self.york, 'traditional')
            self.york_dl = os.path.join(self.york, 'dl')
            self.sheffield = os.path.join(self.anonymised, 'sheffield')
            self.cpet_db = os.path.join(
                self.data_path, 'CPETdb.xlsx')
            self.linked_data = os.path.join(
                self.data_path, 'linked data.csv')
            self.logs = "./logs/"
            self.linked_data_with_db = os.path.join(
                self.data_path, 'linked data with db.csv')
            self.anonymised_linked_data_with_db = os.path.join(
                self.data_path, './anonymised/linked data with db.csv')
            self.sum_features = "./sum_features.txt"
            self.york_flat = os.path.join(
                self.york_traditional, 'flat_output_final.csv')
            self.york_binned_normalised = os.path.join(
                self.york_dl, 'binned_normalised')
            self.sheffield_flat = os.path.join(
                self.sheffield, 'flat_output_final.csv')
            self.missing_bxb = os.path.join(self.data_path, 'missing_bxb.txt')
            self.missing_phase_3 = os.path.join(
                self.data_path, 'missing_phase_3.txt')
            self.missing_rpm = os.path.join(self.data_path, 'missing_rpm.txt')
            self.missing_no_reason = os.path.join(
                self.data_path, 'missing_no_reason.txt')
            self.missing_files = os.path.join(
                self.data_path, 'missing_files.txt')

            self.gxt_features = self._initialize_gxt_features()
            self._feature_maps = self._initialize_feature_maps()
            self._spawn_directories()

    def __str__(self):
        return "ProjectStrings class"

    def _spawn_directories(self):
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
        if not os.path.exists(self.cpet_data):
            os.mkdir(self.cpet_data)
        if not os.path.exists(self.logs):
            os.mkdir(self.logs)
        if not os.path.exists(self.anonymised):
            os.mkdir(self.anonymised)
        if not os.path.exists(self.york):
            os.mkdir(self.york)
        if not os.path.exists(self.york_traditional):
            os.mkdir(self.york_traditional)
        if not os.path.exists(self.york_dl):
            os.mkdir(self.york_dl)
        if not os.path.exists(self.york_binned_normalised):
            os.mkdir(self.york_binned_normalised)

    def _initialize_gxt_features(self):
        feature_maps = {}
        start_reading = False
        with open(os.path.join(self.data_path, './template_sum.template'), 'r') as file:
            for line in file:
                if '$;1000;GXTestDataSection;' in line:
                    start_reading = True
                    continue

                if '$;3999;GXTestDataSectionEnd;;' in line:
                    break  # Stop processing after this line

                if start_reading:
                    match = re.search(r'\$;(\d+);([^;]+);', line)
                    if match:
                        code, name = match.groups()
                        if name not in feature_maps.values():
                            feature_maps[name] = f"$;{code};{name};"
        return feature_maps

    def _initialize_feature_maps(self):
        feature_maps = {}
        with open(os.path.join(self.data_path, './template_sum.template'), 'r') as file:
            for line in file:
                if '$;999;PFTestDataSectionEnd;;' in line:
                    break  # Stop processing after this line

                match = re.search(r'\$;(\d+);([^;]+);', line)
                if match:
                    code, name = match.groups()
                    if name not in feature_maps.values():
                        feature_maps[name] = f"$;{code};{name};"
        # GXT features
        feature_maps['HR BPM MaxValue'] = "$;3068;HR BPM"
        feature_maps['HR BPM PredMax'] = "$;3068;HR BPM"
        feature_maps['Chronotropic Index'] = "$;3903;Chronotropic Index"
        feature_maps.update(self._initialize_gxt_features())

        with open('options.txt', 'w') as file:
            for feature in feature_maps:
                file.write(f"{feature}\n")
        return feature_maps

    @property
    def feature_maps(self):
        return self._feature_maps

    @property
    def basic_feature_maps(self):
        # New dictionary to store the extracted values
        new_dict = {}

        # Iterate over each key-value pair in the original dictionary
        for key, value in self.feature_maps.items():
            # Split the value by ';'
            parts = value.split(';')
            if len(parts) > 1:
                # Extract the number between the first two semicolons
                number = parts[1]
                # Update the new dictionary with the key and extracted number
                new_dict[key] = number

        # Print the new dictionary to see the result

        # add some manual mappings
        new_dict['Breath'] = '115'

        new_dict['VO2_kg_extrap mL/kg/min'] = '1048'
        new_dict['VO2WorkSlope mL/min/watt'] = '1124'
        new_dict['DeltaVO2Watts L/Min/Watt'] = '1105'
        new_dict['Speed_RPM RPM'] = '1132'
        new_dict['ExerTime_sec sec'] = '1195'
        return new_dict

    def wanted_feature_maps(self, features: list):
        # validate the features
        for feature in features:
            if feature not in self.feature_maps:
                raise ValueError(
                    f"Feature {feature} not found in feature maps")
        return {feature: self.feature_maps[feature] for feature in features}

    @property
    def required_cpet_db_columns(self):
        """
        Returns the columns of the CPETdb.xlsx file that are expected to be present
        """
        return ["Research number", "patient ID_CPETdb", "hospital number", "case note number 1", "case note number 2", "Date of Birth", "Test Date"]

    @property
    def required_cpet_data_not_nan_columns(self):
        """
        Returns the columns of the CPET data that are expected to be present and not NaN
        """
        return ["Date of Birth", "Test Date"]
