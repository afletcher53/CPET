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

    def _initialize_feature_maps(self):

        feature_maps = {
            # "Age": "$;6025;Age;",
            # "BMI": "$;6100;BMI;",
            # "HR BPM MaxValue": "$;3068;HR BPM",
            # "HR BPM PredMax": "$;3068;HR BPM",
            # "Chronotropic Index": "$;3903;Chronotropic Index",
            # "Height": "$;6020;Height;",
            # "Weight": "$;6021;Weight;", 
            # "Race": "$;6010;Race;",
            # "Hospital Site": "$;6015;Site;",
            # "Start Exercise": "$;6096;StartExercise;"
        }
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
        with open('options.txt', 'w') as file:
            for feature in feature_maps:
                file.write(f"{feature}\n")
        return feature_maps
   
    @property
    def feature_maps(self):
        return self._feature_maps

    
    def wanted_feature_maps(self, features: list):
        # validate the features
        for feature in features:
            if feature not in self.feature_maps:
                raise ValueError(f"Feature {feature} not found in feature maps")
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
