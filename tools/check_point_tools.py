import json
import os
import pickle
from tranforms import Transform


class CheckPoint:
    dir_check = 'Data/Info/Checkpoint'
    dir_descript = 'Data/Info/Descript'

    def __init__(self, name, dir_descript=None):
        self.dir_check = os.path.join(self.dir_check, name + '.txt')
        self.dir_descript = os.path.join(dir_descript,name + '.json') if not (dir_descript is None) else os.path.join(self.dir_descript,
                                                                                         name + '.json')

    def load(self):
        if os.path.exists(self.dir_check):
            with open(self.dir_check, 'rb') as file:
                check = pickle.load(file)
            return check
        else:
            raise FileNotFoundError

    def exit(self):
        return os.path.exists(self.dir_descript)

    def save(self, transform: Transform):

        with open(self.dir_descript, 'w') as file:
            json.dump(transform.info(), file, indent=4)

        with open(self.dir_check, 'wb') as file:
            pickle.dump(transform, file)
