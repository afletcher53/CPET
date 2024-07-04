import os


class ProjectStrings:
    _shared_state = {}

    def __init__(self):
        self.__dict__ = self._shared_state
        if not self._shared_state:
            self.data_path = "./data/"
            self.cpet_data = os.path.join(self.data_path, 'cpet raw')
            self.cpet_db = os.path.join(
                self.data_path, 'CPETdb.xlsx')
            self.linked_data = os.path.join(
                self.data_path, 'linked data.csv')
            self.logs = "./logs/"
            self.linked_data_with_db = os.path.join(
                self.data_path, 'linked data with db.csv')
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
