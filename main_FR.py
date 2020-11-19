import argparse
import time
from ctypes import c_uint8
from multiprocessing import Process, Queue, Value

import cv2
from imutils.video import FileVideoStream, WebcamVideoStream

from lib.server import save, server_send, temp_check
from lib.tflite import Detector, Recognizer
from lib.track import Tracker, image_encode  # noqa
from lib.utils import ConfigHandler, draw


def parse_arg():
    ap = argparse.ArgumentParser()
    ap.add_argument('-s', '--source', default='vid', choices=('cam', 'vid'),
                    help='Select input source, "cam" or "vid"')
    ap.add_argument('-v', '--vid-path', default='test/videos/test2.m4v',
                    help='Path to video input when source is "vid"')
    ap.add_argument('-d', '--direction', required=True, choices=('in', 'out'),
                    help='Camera tracking direction "in" or "out"')
    args = vars(ap.parse_args())
    return args


def init_constant():
    global config, src, stream, detector, recognizer, args, tracker
    source = args['source']
    dir_ = args['direction'].lower()

    if source == 'cam':
        VS = WebcamVideoStream
        src = config.stream[dir_]
    else:
        VS = FileVideoStream
        src = args['vid_path']
    stream = VS(src).start()

    detector = Detector(config.path['detect_model'],
                        config.model_setting['min_face_HD'],
                        config.model_setting['threshold'],
                        config.model_setting['face_size'])
    recognizer = Recognizer(config.path['recog_model'],
                            config.path['labels'])
    tracker = Tracker(dir_, config.tracker['min_dist'][dir_],
                      config.tracker['min_appear'][dir_],
                      config.tracker['max_disappear'][dir_])


def main(img_queue, temp):
    def stop():
        temp.value = 0
        stream.stop()
        img_queue.put('stop')
        file.close()

    init_constant()
    counter = 0
    file = open('log/time_log.txt', 'a')
    file.write(time.strftime('# %d.%m\n'))
    while True:
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
        # -------------------------------MAIN-1------------------------------ #
        if counter < 10:
            counter += 1
        elif counter == 10:
            with open('log/working', 'w') as f:
                f.write('true')
        boxes, faces = detector.detect(frame, True)
        if config.oper['mode']:
            preds = recognizer.recognize(faces)
            objs, datas = tracker.track(boxes, preds, faces)
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
            for data in datas:
                if img_queue.qsize() >= 126:
                    save(data, img_queue)
                else:
                    img_queue.put(data)
                    file.write(time.strftime('%H:%M\n'))
            if config.oper['display']:
                draw(frame, boxes, names)
                cv2.imshow('frame', cv2.resize(frame, (720, 540)))
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    stop()
                    break
        # ------------------------------------------------------------------- #
        # -------------------------------MAIN-2------------------------------ #
        # if config.oper['mode'] < 2:
        #     # --------------------------------------------------------------- #
        #     # -----------------------FRAME SKIP COUNTER---------------------- #

        #     if counter % config.oper['frame_per_capture'] != 0:
        #         continue
        #     counter = 0
        #     for face in faces:
        #         data = {
        #             'timestamp': int(time.time()),
        #             'camera': args['direction'],
        #             'name': '',
        #             'capture': image_encode(face)
        #         }
        #         if img_queue.qsize() >= 126:
        #             save(data)
        #         else:
        #             img_queue.put(data)


if __name__ == "__main__":
    config = ConfigHandler().read()
    args = parse_arg()
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
