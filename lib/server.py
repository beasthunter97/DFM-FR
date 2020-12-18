import os
import time
from subprocess import PIPE, Popen

import requests
from requests import ConnectionError

from lib.utils import name_gen


def server_send(data_queue, config):
    stop = False
    while True:
        if data_queue.empty():
            data = load()
            if data is None:
                if stop:
                    break
                continue
        else:
            data = []
            if data_queue.qsize() < config.oper['max_item']:
                time.sleep(0.5)
            for _ in range(int(min(data_queue.qsize(), config.oper['max_item']))):
                dat = data_queue.get()
                if dat == 'stop':
                    stop = True
                    continue
                data.append(dat)
        server_time = time.time()
        try:
            response = requests.post(config.url['capture'], json=data, verify=False)
            if response.status_code == 200:
                server_status = 'Success'
            else:
                server_status = 'Error ' + str(response.status_code)
        except ConnectionError:
            server_status = 'No internet connection'
        server_time = time.time() - server_time
        print('[SERVER] Time cost %6.2f second(s) | Status: %s' %
              (server_time, server_status))
        if server_status == 'Success':
            continue
        save(data)
        time.sleep(2)


def temp_check(temp, config):
    while True:
        time.sleep(config.oper['time_check_temp'])
        if not temp.value:
            break
        else:
            out = Popen(['cat', '/sys/class/thermal/thermal_zone0/temp'],
                        stdout=PIPE).communicate()[0]

        temp.value = int(out.decode("utf-8").split('000')[0])
        data = {
            "nom": "In",
            "temperature": temp.value,
            "timestamp": int(time.time()) + 7*3600,
            "status": 1,
        }
        try:
            response = requests.post(config.url['status'], data=data, verify=False)
            if response.status_code == 200:
                if temp.value > config.oper['max_temp']:
                    device_status = 'Overheated (%d)' % temp.value
                else:
                    device_status = 'Normal (%d)' % temp.value
            else:
                device_status = 'Error ' + str(response.status_code)
        except ConnectionError:
            device_status = 'No internet connection'
        print('[DEVICE] Status: ', device_status)


def save(data, queue, max_face=10):
    if isinstance(data, list):
        file_name = name_gen(16)
        with open(file_name, 'w') as file:
            file.write(str(data))
    else:
        data = [data]
        for _ in range(max_face):
            data.append(queue.get())
        save(data)


def load():
    retval = None
    for root, dirs, files in os.walk('temp/'):
        if files == []:
            break

        for file in files.sort():
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
