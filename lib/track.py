from scipy.spatial import distance # noqa
from lib.utils import image_encode
import numpy as np # noqa
import time


class Tracker:
    def __init__(self):
        self.old = []

    def track(self, boxes, predictions, faces):
        data_format = """{
            "timestamp": "%s",
            "camera": "%s",
            "name": "%s",
            "capture": %r
            }"""
        datas = []
        names = []
        for box, prediction, face in zip(boxes, predictions, faces):
            data = eval(data_format % (time.strftime('%Y.%m.%d_%H.%M.%S'),
                                       'out',
                                       prediction,
                                       image_encode(face)))
            datas.append(data)
            names.append(prediction)
        return names, datas
