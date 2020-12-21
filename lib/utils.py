import os
import random
import string
import sys
import time

import cv2
import yaml


class ConfigHandler:
    """
    Config Handler class.
    =====================

    Read & write config file.
    """
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


def draw(image, boxes, names, in_out=None):
    for name, (x1, y1, x2, y2) in zip(names, boxes):
        cv2.rectangle(image, (x1, y1), (x2, y2),
                      (0, 255, 0), 2)
        cv2.putText(image, name, (x1 - 15 if x1 - 30 > 0 else x1 + 15,
                                  y1 - 15 if y1 - 30 > 0 else y1 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if in_out is not None:
        text = 'COUNT: {}'.format(*in_out)
        cv2.putText(image, text, (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


def name_gen(length=16, root='temp/', ext='', rand=False):
    os.makedirs(root, exist_ok=True)
    character = string.ascii_letters + string.digits
    while True:
        if rand:
            file_name = root + \
                ''.join(random.choice(character) for _ in range(length))
        else:
            i = 0
            file_name = root + str(int(time.time()))
        if os.path.exists(file_name + ext):
            if rand:
                continue
            else:
                while os.path.exists(file_name + '_%d' % i + ext):
                    i += 1
        break
    return file_name + ext


def get_size(obj, seen=None) -> int:
    """
    Recursively find size of an object.

    Args:
        obj ([type]): Input object.
        seen ([type], optional): Size of the object is counted or not. Defaults to None.

    Returns:
        int: Size of the input object.
    """
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def save(data):
    file_name = name_gen(16)
    with open(file_name, 'w') as file:
        file.write(str(data))


def load():
    retval = None
    for root, dirs, files in os.walk('temp/'):
        if files == []:
            break
        files.sort()
        for file in files:
            file = '/'.join((root, file))
            try:
                with open(file, 'r') as f:
                    data = f.read()
                    retval = eval(data)
                os.remove(file)
                break
            except: # noqa
                os.remove(file)
                continue
        break
    return retval
