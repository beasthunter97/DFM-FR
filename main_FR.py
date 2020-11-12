import argparse
import time
from ctypes import c_int
from multiprocessing import Process, Queue, Value

import cv2
from imutils.video import FileVideoStream, WebcamVideoStream

from lib.server import server_send, temp
from lib.tflite import Detector, Recognizer
from lib.track import Tracker, image_encode
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
                        config.model_setting['threshold'])
    recognizer = Recognizer(config.path['recog_model'],
                            config.path['labels'])
    tracker = Tracker(dir_, config.tracker['min_dist'][dir_],
                      config.tracker['min_appear'][dir_],
                      config.tracker['max_disappear'][dir_])


def main(img_queue, temper):
    init_constant()
    counter = 0
    while True:
        # -------------------------CHECK TEMPERATURE------------------------- #
        if temper.value > config.oper['max_temp']:
            print('Overheated, sleep for 5 seconds')
            time.sleep(config.oper['overheated_sleep'])
            temper.value = 0
        # ------------------------------------------------------------------- #
        # -------------------------READ & CHECK FRAME------------------------ #
        frame = stream.read()
        if frame is None:
            stream.stop()
            img_queue.put('stop')
            break
        # ------------------------------------------------------------------- #
        # -------------------------------MAIN-1------------------------------ #
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
                    temp(data, img_queue)
                else:
                    img_queue.put(data)
            if config.oper['dislay']:
                draw(frame, boxes, names)
                cv2.imshow('frame', cv2.resize(frame, (720, 540)))
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    stream.stop()
                    img_queue.put('stop')
                    break
        # ------------------------------------------------------------------- #
        # -------------------------------MAIN-2------------------------------ #
        if config.oper['mode'] < 2:
            # --------------------------------------------------------------- #
            # -----------------------FRAME SKIP COUNTER---------------------- #
            counter += 1
            if counter % config.oper['frame_per_capture'] != 0:
                continue
            counter = 0
            for face in faces:
                img_queue.put({
                    'timestamp': int(time.time()),
                    'camera': args['direction'],
                    'name': '',
                    'capture': image_encode(face)
                })


if __name__ == "__main__":
    config = ConfigHandler().read()
    args = parse_arg()
    img_queue = Queue(maxsize=128)
    temper = Value(c_int)
    temper.value = 0
    main_process = Process(target=main,
                           args=(img_queue, temper,), name='Main')
    server_process = Process(target=server_send,
                             args=(img_queue, temper, config,), name='Server')
    server_process.start()
    main_process.start()
    main_process.join()
    server_process.join()
