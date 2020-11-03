import base64
import os
import random
import string
import time

import cv2
import requests
import yaml
from requests.exceptions import ConnectionError


class ConfigHandler:
    def __init__(self, config_path='config.yml'):
        self.file_path = config_path

    def read(self, keywords=[]):
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


def server_send(img_queue, url, method='post'):
    while True:
        from_temp = True
        if img_queue.empty():
            data = load_temp()
            if data is None:
                time.sleep(1)
                continue
        else:
            data = img_queue.get()
            from_temp = False
        start_time = time.time()
        print("Queue received")
        if data == 'stop':
            break
        try:
            respond = requests.post(url, data=data, verify=False)
            if respond.status_code == 200:
                print('Success')
            elif respond.status_code == 429:
                time.sleep(5)
                if not from_temp:
                    img_queue.put(data)
                else:
                    temp(data)
            else:
                print(respond.status_code)
                if not from_temp:
                    img_queue.put(data)
                else:
                    temp(data)
        except ConnectionError:
            print('Check internet connection.')
            if not from_temp:
                img_queue.put(data)
            else:
                temp(data)
        print('Time cost %.2f second(s)' % (time.time() - start_time))


def image_encode(img):
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer)


def random_name(length=16):
    os.makedirs('temp/', exist_ok=True)
    character = string.ascii_letters + string.digits
    while True:
        file_name = 'temp/' + ''.join(random.choice(character) for _ in range(length))
        if os.path.exists(file_name):
            continue
        break
    return file_name


def temp(data):
    file_name = random_name(16)
    with open(file_name, 'w') as file:
        file.write(str(data))


def load_temp():
    retval = None
    for root, dirs, files in os.walk('temp/'):
        if files == []:
            break
        else:
            file_name = '/'.join((root, files[0]))
            with open(file_name, 'r') as file:
                data = file.read()
                retval = eval(data)
            os.remove(file_name)
    return retval
