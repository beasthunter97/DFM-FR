import base64
import time

import cv2
import numpy as np
from scipy.spatial import distance


def image_encode(img: np.ndarray) -> str:
    """
    Encode image to string before export to data and sent to server.

    Args:
        img (np.ndarray): Input image.

    Returns:
        str: Encoded string of image.
    """
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')


class Tracker:
    """
    Face tracking class.
    ====================

    Track, count and stack identities.
    """
    def __init__(self, direction: str, max_ratio: float, min_appear: int, skip_frame: int,
                 max_disappear: int, max_img_stack: int, max_send: int):
        """
        Class initialize.

        Args:
            direction (str): Tracking direction, either ``in`` or ``out``.
            max_ratio (float): Max face ratio between two frame.
            min_appear (int): Minimum appearing frames to register event.
            max_disappear (int): Maximum disappearing frames to delete identity and
                                 export event.
            max_img_stack (int): Maximum face image to be stored.
            skip_frame (int): Number of frames to skip before store a new image.
            max_send (int): Maximum number of imagesS to be exported to server.
        """
        self.dir = direction.capitalize()
        self.max_ratio = max_ratio
        self.min_appear = min_appear
        self.max_disappear = max_disappear
        self.obj = []
        self.in_out = [0]
        self.max_stack = max_img_stack
        self.skip = skip_frame
        self.max_send = max_send
        try:
            with open('log/unknown', 'r') as file:
                self.unknown = int(file.read())
                if self.unknown >= 10000:
                    self.unknown = 0
        except FileNotFoundError:
            self.unknown = 0

    def track(self, boxes: list, preds: list, faces: list) -> tuple:
        """
        Track faces on new frame.

        Args:
            boxes (list): Face bounding boxes.
            preds (list): Face recognized names and probabilities.
            faces (list): Face images.

        Returns:
            tuple: tuple of current frame's `face infomation`, `exported data`,
                   and ``in_out`` `count number`.
        """
        self.new_obj = []
        self.data = {}
        self.create_obj(boxes, preds, faces)
        self.update()
        return self.new_obj, self.data, self.in_out

    def create_obj(self, boxes: list, preds: list, faces: list):
        """
        Create new tracking object.

        Args:
            boxes (list): Face bounding boxes.
            preds (list): Face recognized names and probabilities.
            faces (list): Face images.
        """
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            pos = [(x2 + x1)//2, (y2 + y1)//2]
            self.new_obj.append({
                'faces': np.array([faces[i]]),
                'pos': pos,
                'id': np.random.randint(100),
                'name': self.get_true_names(preds[i]),
                'size': x2-x1,
                'pred': preds[i],
                'dir': self.dir
            })

    def update(self):
        """
        Tracking and update current object from new object.
        """
        if self.obj == []:
            for new in range(len(self.new_obj)):
                self.update_obj(new=new)
        elif self.new_obj == []:
            for old in range(len(self.obj)-1, -1, -1):
                self.update_obj(old=old)
        else:
            pos = [i['pos'] for i in self.obj]
            new_pos = [i['pos'] for i in self.new_obj]
            dist = distance.cdist(pos, new_pos)
            old_ = []
            new_ = []
            for _ in range(min(dist.shape)):
                min_pos = dist.argmin()
                min_new = min_pos % dist.shape[1]
                min_old = min_pos // dist.shape[1]
                face_ratio = self.new_obj[min_new]['size']/self.obj[min_old]['size']
                if face_ratio < 1:
                    face_ratio = 1 - face_ratio
                else:
                    face_ratio = 1 - 1/face_ratio
                if face_ratio > self.max_ratio:
                    dist[min_old, min_new] = 2000
                    continue
                old_.append(min_old)
                new_.append(min_new)
                dist[min_old, :] = 2000
                dist[:, min_new] = 2000
            for new, old in zip(new_, old_):
                self.update_obj(new=new, old=old)
            for old in range(dist.shape[0]-1, -1, -1):
                if old not in old_:
                    self.update_obj(old=old)
            for new in range(dist.shape[1]):
                if new not in new_:
                    self.update_obj(new=new)

    def update_obj(self, new=None, old=None):
        """
        Update a specific object.
        > If both ``new`` and ``old`` is `not` ``None``. The new object is updated
        > to be the new status of the corresponding old object.

        > If ``new`` is `not` ``None`` and ``old`` is ``None``. The new object is
        > assigned as a additional current object.

        > If ``new`` is ``None`` and ``old`` is `not` ``None``. The old object is
        > assumed disappeared. If the old object disappeared for too many frames, it
        > will be exported.

        Args:
            new (list, optional): Index list of new object need to be updated.
                                  Defaults to None.
            old (list, optional): Index list of current object need to be updated.
                                  Defaults to None.
        """
        # New obj and old obj is the same obj
        if old is not None and new is not None:
            for name in self.obj[old]['pred']:
                if name in self.new_obj[new]['pred']:
                    self.new_obj[new]['pred'][name] += self.obj[old]['pred'][name]
                else:
                    self.new_obj[new]['pred'][name] = self.obj[old]['pred'][name]
            if self.obj[old]['appear'] % self.skip == 0:
                self.new_obj[new]['faces'] = np.append(self.new_obj[new]['faces'],
                                                       self.obj[old]['faces'], 0)
            else:
                self.new_obj[new]['faces'] = self.obj[old]['faces']
            if len(self.obj[old]['faces']) > self.max_stack:
                indices = list(range(self.max_stack-1)) + [-1]
                self.new_obj[new]['faces'] = self.new_obj[new]['faces'][indices]
            self.new_obj[new]['id'] = self.obj[old]['id']
            self.new_obj[new]['name'] = self.get_true_names(self.new_obj[new]['pred'])
            self.obj[old].update(self.new_obj[new])
            self.obj[old]['appear'] += 1
            self.obj[old]['disappear'] = 0
        # Existed obj is not in current frame
        elif old is not None:
            self.obj[old]['disappear'] += 1
            if self.obj[old]['disappear'] > self.max_disappear:
                obj = self.obj.pop(old)
                if obj['appear'] > self.min_appear:
                    self.export_obj(obj)
        # New obj appear
        elif new is not None:
            self.obj.append(self.new_obj[new])
            self.obj[-1].update({
                'size_0': self.new_obj[new]['size'],
                'appear': 1,
                'disappear': 0
            })

    def export_obj(self, obj: dict):
        """
        Export object to server.

        Args:
            obj (dict): Object to be exported
        """
        self.in_out[0] += 1
        if 'UNKNOWN' == obj['name']:
            obj['name'] = '%s-%d' % (obj['name'], self.unknown)
            with(open('log/unknown', 'w')) as file:
                file.write(str(self.unknown))
            self.unknown += 1
            index = range(0, -max(self.max_send, len(obj['faces'])), -1)
            capture = [image_encode(obj['faces'][i]) for i in index]
        else:
            capture = [image_encode(obj['faces'][-1])]
        self.data = {
            'timestamp': int(time.time() + 7 * 3600),
            'camera': obj['dir'],
            'name': obj['name'],
            'capture': capture
        }

    def get_true_names(self, preds: dict) -> str:
        """
        Return name of the identity from the predicted ``name``: ``confidence``.

        Args:
            preds (dict): Dictionary of ``name``: ``confidence``.

        Returns:
            str: Name of the highest confidence.
        """
        conf = max(preds.values())
        if conf < 0.5:
            return 'UNKNOWN'
        key = [k for k, v in preds.items() if v == conf][0]
        return key
