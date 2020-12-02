import time
from ctypes import c_uint8
from multiprocessing import Process, Queue, Value

import cv2
from imutils.video import FileVideoStream, WebcamVideoStream

from lib.server import save, server_send, temp_check
from lib.tflite import Detector, Recognizer
from lib.track import Tracker, image_encode
from lib.utils import ConfigHandler, draw


def init_constant():
    global config, src, stream, detector, recognizer, tracker
    source = config.source['src']
    dir_ = config.source['direction']

    if source == 'cam':
        VS = WebcamVideoStream
        src = config.stream[dir_]
    else:
        VS = FileVideoStream
        src = config.source['vid_path']
    stream = VS(src).start()
    detector = Detector(config.path['detect_model'],
                        config.model_setting['min_face_HD'],
                        config.model_setting['threshold'],
                        config.model_setting['face_size'])
    recognizer = Recognizer(config.path['recog_model'],
                            config.path['labels'],
                            config.oper['mode'])
    tracker = Tracker(dir_, config.tracker['min_dist'][dir_],
                      config.tracker['min_appear'][dir_],
                      config.tracker['max_disappear'][dir_],
                      config.oper['mode'])


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
        # ---------------------------CHECK WORKING--------------------------- #
        print(counter)
        if counter < 10:
            counter += 1
        elif counter == 10:
            counter = 11
            with open('log/working', 'w') as f:
                f.write('true\n')
        # ------------------------------------------------------------------- #
        # -------------------------------TRACK------------------------------- #
        boxes, faces = detector.detect(frame, True)
        if config.oper['mode']:
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
        if config.oper['mode'] != 2 and counter > 10:
            if counter % config.oper['frame_per_capture'] != 0:
                counter += 1
            else:
                counter = 11
                for face in faces:
                    datas.append({
                        'timestamp': int(time.time()),
                        'camera': 'data',
                        'name': '',
                        'capture': image_encode(face)
                    })
        for data in datas:
            if img_queue.qsize() >= 120:
                save(data, img_queue)
            else:
                img_queue.put(data)
                file.write(time.strftime('%H:%M\n'))
        # ------------------------------------------------------------------- #
        # ------------------------------DISPLAY------------------------------ #
        if config.oper['display']:
            draw(frame, boxes, names, in_out)
            cv2.imshow('frame', cv2.resize(frame, (720, 540)))
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                stop()
                break


if __name__ == "__main__":
    config = ConfigHandler().read()
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
