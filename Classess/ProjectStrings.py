import os

class ProjectStrings:
    _shared_state = {}

    def __init__(self):
        self.__dict__ = self._shared_state
        if not self._shared_state:
            self.data_path = "./data/"
            self.cpet_data = os.path.join(self.data_path, 'cpet raw')

    def __str__(self):
        return "ProjectStrings class"
