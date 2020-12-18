"""
This file is execute by the ``boot_script.sh``.

It contains of the main and sub processes.
"""
# region IMPORT
import time
from ctypes import c_uint8
from multiprocessing import Process, Queue, Value
from subprocess import PIPE, Popen

import cv2
from imutils.video import FileVideoStream, WebcamVideoStream
from requests import ConnectionError, post

from lib.tflite import Detector, Recognizer
from lib.track import Tracker
from lib.utils import ConfigHandler, draw, load, save
# endregion


def server_process(data_queue):
    stop = False
    while True:
        if data_queue.empty():
            data = load()
            if data is None:
                if stop:
                    break
                continue
        else:
            data = data_queue.get()
            if data == 'stop':
                stop = True
                continue
        server_time = time.time()
        try:
            response = post(config.server['url'], json=data, verify=False)
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
        time.sleep(config.server['time_out'])


def temp_process(temp):
    while True:
        time.sleep(config.temp['time_check_temp'])
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
            response = post(config.temp['url'], json=data, verify=False)
            if response.status_code == 200:
                if temp.value > config.temp['max_temp']:
                    device_status = 'Overheated (%d)' % temp.value
                else:
                    device_status = 'Normal (%d)' % temp.value
            else:
                device_status = 'Error ' + str(response.status_code)
        except ConnectionError:
            device_status = 'No internet connection'
        print('[DEVICE] Status: ', device_status)


def main_process(data_queue: 'Queue', temp: 'c_uint8') -> None:
    """
    ``Main`` process.
    Performs `detection`, `recognition` and `tracking` from video stream and send
    infomation to ``Server`` process.

    Args:
        data_queue (Queue): Shared variable with ``Server`` process. The ``Main``
                            process put data to it and the ``Server`` process will
                            send receive it and send it to `DFM Server`
        temp (c_uint8): Shared variable with ``Temp`` process. The ``Main`` process
                        read device's temperature from this variable.
    """
    def stop() -> None:
        """
        Necessary procedure to stop the program.
        """
        temp.value = 0
        stream.stop()
        data_queue.put('stop')

    def init_constant(config) -> None:
        """
        Initialize constants and objects for the main process.
        """
        global src, stream, detector, recognizer, tracker
        direction = config.direction
        tracking = config.tracking['shared']
        tracking.update(config.tracking[direction])
        src = config.source[direction]
        if config.source['type'] == 'cam':
            VS = WebcamVideoStream
        else:
            VS = FileVideoStream
        stream = VS(src).start()
        detector = Detector(**config.detection)
        recognizer = Recognizer(**config.recognition)
        tracker = Tracker(direction, **tracking)

    init_constant(config)
    counter = 0
    with open('log/time_log.txt', 'a') as file:
        file.write(time.strftime('# %d.%m\n'))
    while True:
        # region CHECK TEMPERATURE
        if temp.value > config.temp['max_temp']:
            print('Overheated, sleep for %d seconds' % config.oper['overheated_sleep'])
            time.sleep(config.temp['overheated_sleep'])
            temp.value = 1
        # endregion

        # region READ & CHECK FRAME
        frame = stream.read()
        if frame is None:
            stop()
            break
        # endregion

        # region CHECK WORKING
        if counter < 10:
            counter += 1
        elif counter == 10:
            counter += 1
            with open('log/working', 'w') as f:
                f.write('true\n')
        # endregion

        # region DETECT, RECOGNIZE, TRACK
        boxes, faces = detector.detect(frame, True)
        preds = recognizer.recognize(faces)
        objs, data, in_out = tracker.track(boxes, preds, faces)
        names = []
        boxes = []
        for obj in objs:
            names.append(str(obj['name']))
            pos, size = obj['pos'], obj['size']
            boxes.append([
                pos[0] - size//2,
                pos[1] - size//2,
                pos[0] + size//2,
                pos[1] + size//2
            ])
        # endregion

        # region SEND DATA
        if data != {}:
            with open('log/time_log.txt', 'a') as file:
                file.write(time.strftime('%H:%M\n'))
            if data_queue.qsize() >= 120:
                save(data)
            else:
                data_queue.put(data)
        # endregion

        # region DISPLAY
        if config.display:
            show = frame.copy()
            draw(show, boxes, names, in_out)
            cv2.imshow('frame', cv2.resize(show, (720, 540)))
            key = cv2.waitKey(1) & 0xFF
            del show
            if key == ord('q'):
                stop()
                break
        # endregion


if __name__ == "__main__":
    config = ConfigHandler().read()
    data_queue = Queue(maxsize=128)
    temp = Value(c_uint8)
    temp.value = 1
    main_process = Process(target=main_process,
                           args=(data_queue, temp,), name='Main')
    server_process = Process(target=server_process,
                             args=(data_queue,), name='Server')
    temp_process = Process(target=temp_process,
                           args=(temp,), name='Temp')
    temp_process.start()
    server_process.start()
    main_process.start()
    main_process.join()
    server_process.join()
    temp_process.join()
