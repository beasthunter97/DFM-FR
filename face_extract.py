from imutils.video import FileVideoStream as FVS
from lib.utils import random_name
from lib.tflite import Detector
import os
import cv2


detector = Detector('models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite',
                    96)
dir = '/home/anhnq/Py/data'
file_dirs = []
for root, dirs, files in os.walk(dir):
    for file in files:
        file_dirs.append('/'.join((root, file)))
check = True
i = 0
for file in file_dirs:
    stream = FVS(file).start()
    width, height = stream.stream.get(cv2.CAP_PROP_FRAME_WIDTH),\
        stream.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
    while True:
        frame = stream.read()
        if frame is None:
            stream.stop()
            break
        boxes, faces = detector.detect(frame, True)
        for box, face in zip(boxes, faces):
            file_name = random_name(6, '/home/anhnq/Py/data/face_img/%d' % i,
                                    '.png')
            cv2.imwrite(file_name, face)
            i += 1
