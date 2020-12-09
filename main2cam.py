import time
from ctypes import c_uint8
from multiprocessing import Process, Queue, Value

import cv2
from imutils.video import FileVideoStream, WebcamVideoStream

from lib.server import save, server_send, temp_check # noqa
from lib.tflite import Detector, Recognizer
from lib.track import Tracker
from lib.utils import ConfigHandler, draw # noqa


def init_constant():
    global config, src_1, src_2, stream_1, stream_2, detector, recognizer, tracker_1, tracker_2 # noqa

    if config.source['src'] == 'cam':
        VS = WebcamVideoStream
        src_1 = config.stream['in']
        src_2 = config.stream['out']
    else:
        VS = FileVideoStream
        src_1 = config.source['vid_path']
    stream_1 = VS(src_1).start()
    stream_2 = VS(src_2).start()

    detector = Detector(config.path['detect_model'],
                        config.model_setting['min_face_HD'],
                        config.model_setting['threshold'],
                        config.model_setting['face_size'])
    recognizer = Recognizer(config.path['recog_model'],
                            config.path['labels'])
    tracker_1 = Tracker(config)
    tracker_2 = Tracker(config)


def main(img_queue, temp):
    def stop():
        temp.value = 0
        stream_1.stop()
        stream_2.stop()
        img_queue.put('stop')

    init_constant()
    counter = 0
    with open('log/time_log.txt', 'a') as file:
        file.write(time.strftime('# %d.%m\n'))
    cv2.namedWindow('frame')
    cv2.moveWindow('frame', 20, 20)
    while True:
        start_time = time.time()
        # ------------------------------------------------------------------- #
        # -------------------------CHECK TEMPERATURE------------------------- #
        if temp.value > config.oper['max_temp']:
            print('Overheated, sleep for 5 seconds')
            time.sleep(config.oper['overheated_sleep'])
            temp.value = 1
        # ------------------------------------------------------------------- #
        # -------------------------READ & CHECK FRAME------------------------ #
        frame_1 = stream_1.read()
        if frame_1 is None:
            stop()
            break
        frame_2 = stream_2.read()
        if frame_2 is None:
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
        boxes, faces = detector.detect(frame_1, True)
        preds = recognizer.recognize(faces)
        objs, datas, in_out = tracker_1.track(boxes, preds, faces)
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
        boxes, faces = detector.detect(frame_2, True)
        preds = recognizer.recognize(faces)
        objs, datas, in_out = tracker_2.track(boxes, preds, faces)
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
        # for data in datas:
        #     with open('log/time_log.txt', 'a') as file:
        #         file.write(time.strftime('%H:%M\n'))
        #     if img_queue.qsize() >= 120:
        #         save(data, img_queue)
        #     else:
        #         img_queue.put(data)
        # ------------------------------------------------------------------- #
        # ------------------------------DISPLAY------------------------------ #
        if config.oper['display']:
            show = frame_1.copy()
            # draw(show, boxes, names, in_out)
            cv2.imshow('frame1', cv2.resize(show, (720, 540)))
            key = cv2.waitKey(1) & 0xFF
            del show
            if key == ord('q'):
                stop()
                break
        if config.oper['display']:
            show = frame_2.copy()
            # draw(show, boxes, names, in_out)
            cv2.imshow('frame2', cv2.resize(show, (720, 540)))
            key = cv2.waitKey(1) & 0xFF
            del show
            if key == ord('q'):
                stop()
                break
        print(time.time() - start_time)


if __name__ == "__main__":
    config = ConfigHandler().read()
    config.source['direction'] = 'in'
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
