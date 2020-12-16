import argparse
import time
from ctypes import c_uint8
from multiprocessing import Process, Queue, Value

import cv2
from imutils.video import FileVideoStream, WebcamVideoStream

from lib.server import save, server_send, temp_check
from lib.tflite import Detector, Recognizer
from lib.track import Tracker
from lib.utils import ConfigHandler, draw


def parse_arg() -> str:
    """
    Parse command line argument. Currently there's only one argument
    needed. So it return that specific argument.

    Returns:
        `str:` direction argument
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--direction', default='out', choices=('in', 'out'),
                    help='Camera tracking direction "in" or "out"')
    args = vars(ap.parse_args())
    return args['direction']


def init_constant() -> None:
    """
    Initialize important/constant objects for the main process.
    """
    global config, src, stream, detector, recognizer, tracker

    if config.source['src'] == 'cam':
        VS = WebcamVideoStream
        src = config.stream[config.source['direction']]
    else:
        VS = FileVideoStream
        src = config.source['vid_path']
    stream = VS(src).start()
    detector = Detector(config.path['detect_model'],
                        config.model_setting['min_face_HD'],
                        config.model_setting['threshold'],
                        config.model_setting['face_size'])
    recognizer = Recognizer(config.path['recog_model'],
                            config.path['labels'])
    tracker = Tracker(config)


def main(img_queue: Queue, temp: c_uint8) -> None:
    """
    <h1>Main process</h1>. Included tasks: stream reading, face detect, face track, face
    recognize, put data to Server process, show frameimport matplotlib.pyplot as plt.

    Args:
        `img_queue` (`Queue`): Image queue to communicate with Server process
        `temp` (`c_uint8`): Temperature variable to communicate with Temp process
    """
    def stop():
        temp.value = 0
        stream.stop()
        img_queue.put('stop')

    init_constant()
    counter = 0
    with open('log/time_log.txt', 'a') as file:
        file.write(time.strftime('# %d.%m\n'))
    cv2.namedWindow('frame')
    cv2.moveWindow('frame', 20, 20)
    while True:
        # ------------------------------------------------------------------- #
        # -------------------------CHECK TEMPERATURE------------------------- #
        if temp.value > config.oper['max_temp']:
            print('Overheated, sleep for 5 seconds')
            time.sleep(config.oper['overheated_sleep'])
            temp.value = 1
        # ------------------------------------------------------------------- #
        # -------------------------READ & CHECK FRAME------------------------ #
        frame = stream.read()
        if frame is None:
            stop()
            break
        # ------------------------------------------------------------------- #
        # ---------------------------CHECK WORKING--------------------------- #
        if counter < 10:
            counter += 1
        elif counter == 10:
            counter += 1
            with open('log/working', 'w') as f:
                f.write('true\n')
        # ------------------------------------------------------------------- #
        # -------------------------------TRACK------------------------------- #
        boxes, faces = detector.detect(frame, True)
        preds = recognizer.recognize(faces)
        objs, datas, in_out = tracker.track(boxes, preds, faces)
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
        # ------------------------------------------------------------------- #
        # -----------------------------SEND-DATA----------------------------- #
        for data in datas:
            with open('log/time_log.txt', 'a') as file:
                file.write(time.strftime('%H:%M\n'))
            if img_queue.qsize() >= 120:
                save(data, img_queue)
            else:
                img_queue.put(data)
        # ------------------------------------------------------------------- #
        # ------------------------------DISPLAY------------------------------ #
        if config.oper['display']:
            show = frame.copy()
            draw(show, boxes, names, in_out)
            cv2.imshow('frame', cv2.resize(show, (720, 540)))
            key = cv2.waitKey(1) & 0xFF
            del show
            if key == ord('q'):
                stop()
                break


if __name__ == "__main__":
    direction = parse_arg()
    config = ConfigHandler().read()
    config.source['direction'] = direction
    img_queue = Queue(maxsize=128)
    temp = Value(c_uint8)
    temp.value = 1
    main_process = Process(target=main,
                           args=(img_queue, temp,), name='Main')
    server_process = Process(target=server_send,
                             args=(img_queue, config,), name='Server')
    temp_process = Process(target=temp_check,
                           args=(temp, config), name='Temp')
    temp_process.start()
    server_process.start()
    main_process.start()
    main_process.join()
    server_process.join()
    temp_process.join()
