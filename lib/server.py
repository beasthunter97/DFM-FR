import os
import subprocess
import time

import requests
from requests import ConnectionError

from lib.utils import random_name


class Server:
    """Handling server procedure
    """
    def __init__(self, url_capture: str, url_status: str,
                 max_temp: int, time_check_temp: int):
        """Initialize

        Args:
            url_capture (str): URL to capture server
            url_status (str): URL to status server
            max_temp (int): Max device's temperature
            time_check_temp (int): Time cycle to check temperature
        """
        self.temp = 0
        self.url_capture = url_capture
        self.url_status = url_status
        self.stop = False
        self.max_temp = max_temp
        self.time_check_temp = time_check_temp
        self.temp_time = time.time()

    def check_device_status(self):
        if time.time() - self.temp_time > self.time_check_temp:
            out = subprocess.Popen(['cat', '/sys/class/thermal/thermal_zone0/temp'],
                                   stdout=subprocess.PIPE).communicate()[0]

            self.temp = int(out.decode("utf-8").split('000')[0])
            data = {
                "temperature": self.temp,
                "timestamp": int(time.time()),
                "status": 1
            }
            try:
                response = requests.post(self.url_status, data=data, verify=False)
                if requests.status_code == 200:
                    if self.temp > self.max_temp:
                        self.device_status = 'Overheated (%d)' % self.temp
                    else:
                        self.device_status = 'Normal (%d)' % self.temp
                else:
                    self.device_status = 'Error ' + str(response.status_codes)
            except ConnectionError:
                self.device_status = 'No internet connection'
            self.temp_time = time.time()
        else:
            self.device_status = None

    def get_data(self, img_queue):
        if img_queue.empty():
            self.data = load_temp()
            if self.data is None:
                if self.stop:
                    return 'break'
                return 'continue'
        else:
            self.data = []
            if img_queue.qsize() < 10:
                time.sleep(0.5)
            for _ in range(int(min(img_queue.qsize(), 10))):
                dat = img_queue.get()
                if dat == 'stop':
                    self.stop = True
                    continue
                self.data.append(dat)

    def server_send(self):
        start_time = time.time()
        try:
            response = requests.post(self.url_capture, json=self.data, verify=False)
            if response.status_code == 200:
                server_status = 'Success'
            elif response.status_code == 429:
                server_status = 'Too many requests'
            else:
                server_status = 'Error ' + str(response.status_code)
        except ConnectionError:
            server_status = 'No internet connection'
        total = time.time() - start_time
        self.server_status = (total, server_status)


def server_send(img_queue, temper, config, method='post'):
    server = Server(config.url['capture'], config.url['status'],
                    config.oper['max_temp'], config.oper['time_check_temp'])
    while True:
        server.check_device_status()
        if server.device_status is not None:
            temper.value = server.temp
            print('[DEVICE] Status: %s' % server.device_status)

        command = server.get_data(img_queue)
        if command == 'continue':
            continue
        elif command == 'break':
            break

        server.server_send()
        print('[SERVER] Time cost %.2f second(s) | Status: %s' % server.server_status)
        if server.server_status[1] == 'Too many requests':
            time.sleep(3)
        if server.server_status[1] != 'Success':
            temp(server.data)
            time.sleep(2)


def temp(data: any, img_queue=None):
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
        try:
            with open(file_name, 'r') as file:
                data = file.read()
                retval = eval(data)
        except: # noqa
            os.remove(file_name)
            continue
        os.remove(file_name)
        break
    return retval
