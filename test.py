import argparse
from multiprocessing import Process, Queue

import cv2 # noqa
from imutils.video import FileVideoStream, WebcamVideoStream

from lib.tflite import Detector, Recognizer
from lib.track import Tracker
from lib.utils import ConfigHandler, draw, server_send, temp # noqa


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
    detector = Detector(config.path['detect_model'])
    recognizer = Recognizer(config.path['recog_model'],
                            config.path['labels'])
    tracker = Tracker()
    if source == 'cam':
        VS = WebcamVideoStream
        src = config.stream[0]
    elif source == 'vid':
        VS = FileVideoStream
        src = args['vid_path']
    else:
        raise ValueError('source must be "cam" or "vid"')
    stream = VS(src).start()


def main(img_queue, config, args):
    init_constant()
    while True:
        try:
            frame = stream.read()
            if frame is None:
                stream.stop()
                img_queue.put('stop')
                break
            boxes, faces = detector.detect(frame, True)
            predidctions = recognizer.recognize(faces)
            names, datas = tracker.track(boxes, predidctions, faces)
            for data in datas:
                if img_queue.qsize() >= 112:
                    temp(data)
                else:
                    img_queue.put(data)
            # draw(frame, boxes, names)
            # cv2.imshow('frame', cv2.resize(frame, (720, 540)))
            # key = cv2.waitKey(1) & 0xFF
            # if key == ord('q'):
            #     stream.stop()
            #     img_queue.put('stop')
            #     break
        except KeyboardInterrupt:
            stream.stop()
            img_queue.put('stop')
            break


if __name__ == "__main__":
    config = ConfigHandler().read()
    args = parse_arg()
    img_queue = Queue(maxsize=128)
    main_process = Process(target=main, args=(img_queue, config, args))
    server_process = Process(target=server_send, args=(img_queue, config.url,))
    server_process.start()
    main_process.start()
    main_process.join()
    server_process.join()
