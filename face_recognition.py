import argparse
import time
from ctypes import c_int
from multiprocessing import Process, Queue, Value

import cv2
from imutils.video import FileVideoStream, WebcamVideoStream

from lib.server import server_send, temp
from lib.tflite import Detector, Recognizer
from lib.track import Tracker
from lib.utils import ConfigHandler, draw


def parse_arg():
    ap = argparse.ArgumentParser()
    ap.add_argument('-s', '--source', required=True,
                    help='Select input source, "cam" or "vid"')
    ap.add_argument('-v', '--vid-path', default='test/videos/test2.m4v',
                    help='Path to video input when source is "vid"')
    args = vars(ap.parse_args())
    return args


def init_constant():
    global config, src, stream, detector, recognizer, args, tracker
    source = args['source']
    detector = Detector(config.path['detect_model'],
                        config.model_setting['min_face_HD'],
                        config.model_setting['threshold'])
    recognizer = Recognizer(config.path['recog_model'],
                            config.path['labels'])
    tracker = Tracker('OUT', 10)
    if source == 'cam':
        VS = WebcamVideoStream
        src = config.stream['out']
    elif source == 'vid':
        VS = FileVideoStream
        src = args['vid_path']
    else:
        raise ValueError('source must be "cam" or "vid"')
    stream = VS(src).start()


def main(img_queue, temper):
    init_constant()
    counter = 0
    while True:
        # -------------------------CHECK TEMPERATURE------------------------- #
        if temper.value > config.oper['max_temp']:
            print('Overheated, sleep for 5 seconds')
            time.sleep(config.oper['sleep'])
            temper.value = 0
        # ------------------------------------------------------------------- #
        # -------------------------READ & CHECK FRAME------------------------ #
        frame = stream.read()
        if frame is None:
            stream.stop()
            img_queue.put('stop')
            break
        # ------------------------------------------------------------------- #
        # -------------------------FRAME SKIP COUNTER------------------------ #
        counter += 1
        if counter % config.oper['frame_per_capture'] != 0:
            counter = 0
            continue
        # ------------------------------------------------------------------- #
        # -------------------------------MAIN-1------------------------------ #
        if config.mode:
            boxes, faces = detector.detect(frame, True)
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
            draw(frame, boxes, names)
            time.sleep(0.05)
            cv2.imshow('frame', cv2.resize(frame, (720, 540)))
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                stream.stop()
                img_queue.put('stop')
                break
        else:
            boxes, faces = detector.detect(frame, True)


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
