import os
import subprocess
import time

import requests
from requests import ConnectionError

from lib.utils import random_name


def check_temp(temper, t_temp, url):
    if time.time() - t_temp > 60:
        out = subprocess.Popen(['cat', '/sys/class/thermal/thermal_zone0/temp'],
                               stdout=subprocess.PIPE).communicate()[0]

        temper.value = int(out.decode("utf-8").split('000')[0])
        print('temp: {}'.format(temper.value))
        data = {
            "bus": 1,
            "temperature": temper.value,
            "timestamp": int(time.time()),
            "status": 1
        }
        try:
            requests.post(url, data=data, verify=False)
        except ConnectionError:
            pass
        return True


def server_send(img_queue, temper, config, method='post'):
    url = config.url['capture']
    url_ = config.url['status']
    stop = False
    t_temp = time.time()
    while True:
        if check_temp(temper, t_temp, url_):
            t_temp = time.time()
        if img_queue.empty():
            data = load_temp()
            if data is None:
                if stop:
                    break
                continue
        else:
            data = []
            if img_queue.qsize() < 10:
                time.sleep(0.5)
            for _ in range(int(min(img_queue.qsize(), 10))):
                dat = img_queue.get()
                if dat == 'stop':
                    stop = True
                    continue
                data.append(dat)
        print("Queue received")
        start_time = time.time()
        try:
            respond = requests.post(url, json=data, verify=False)
            if respond.status_code == 200:
                status = 'Success'
            elif respond.status_code == 429:
                status = 'Too many requests'
            else:
                status = 'Error: ' + respond.status_code
        except ConnectionError:
            status = 'No internet connection'
        print('Time cost %.2f second(s) | Status: %s' % (time.time() - start_time,
                                                         status))
        if status == 'Too many requests':
            time.sleep(5)
        if status != 'Success':
            temp(data)
            time.sleep(2)


def temp(data, img_queue=None):
    if isinstance(data, list):
        file_name = random_name(16)
        with open(file_name, 'w') as file:
            file.write(str(data))
    else:
        data = [data]
        for _ in range(9):
            data.append(img_queue.get())
        temp(data)


def load_temp():
    retval = None
    for root, dirs, files in os.walk('temp/'):
        if files == []:
            break
        file_name = '/'.join((root, files[0]))
        with open(file_name, 'r') as file:
            data = file.read()
            retval = eval(data)
        os.remove(file_name)
        break
    return retval
