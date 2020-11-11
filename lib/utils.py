import os
import random
import string

import cv2
import yaml


class ConfigHandler:
    def read(self, file_path='config.yml', keywords=[]):
        self.file_path = file_path
        if not hasattr(self, 'config'):
            with open(self.file_path, 'r') as file:
                self.config = yaml.full_load(file)
                self.keys = self.config.keys()
        if keywords != []:
            if not isinstance(keywords, list):
                keywords = [keywords]
            retval = []
            for keyword in keywords:
                if keyword in self.keys:
                    retval.append(self.config[keyword])
                else:
                    print('Keyword %s not found.' % keyword)
            return retval
        else:
            for key in self.keys:
                if not hasattr(self, key):
                    setattr(self, key, self.config[key])
            return self

    def write(self, file_path=None):
        if file_path is None:
            file_path = self.file_path
        data = {}
        for key in self.keys:
            data[key] = getattr(self, key)
        write_yaml(file_path, data)


def write_yaml(file_path, data):
    with open(file_path, 'w') as file:
        yaml.dump(data, file)


def draw(image, boxes, names):
    for name, (x1, y1, x2, y2) in zip(names, boxes):
        cv2.rectangle(image, (x1, y1), (x2, y2),
                      (0, 255, 0), 2)
        cv2.putText(image, name, (x1 - 15 if x1 - 30 > 0 else x1 + 15,
                                  y1 - 15 if y1 - 30 > 0 else y1 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)


def random_name(length=16, root='temp/', ext=''):
    os.makedirs(root, exist_ok=True)
    character = string.ascii_letters + string.digits
    while True:
        file_name = root + ''.join(random.choice(character) for _ in range(length)) + ext
        if os.path.exists(file_name):
            continue
        break
    return file_name
