import base64
import time

import cv2
import numpy as np
from scipy.spatial import distance


def image_encode(img):
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')


class Tracker:
    def __init__(self, direction='OUT', min_dist=10, min_appear=3, max_disappear=7):
        self.dir = direction
        self.min_dist = min_dist
        self.min_appear = min_appear
        self.max_disappear = max_disappear
        self.obj = []
        try:
            with open('log/unknown', 'r') as file:
                self.i = int(file.read())
                if self.i >= 1000:
                    self.i = 0
        except FileNotFoundError:
            self.i = 0

    def track(self, boxes, preds, faces):
        self.new_obj = []
        self.datas = []
        self.create_obj(boxes, preds, faces)
        self.update()
        # ------------------------------------------------------------------------ #
        for obj in self.obj:
            obj['name'] = obj['id']
        for obj in self.new_obj:
            obj['name'] = obj['id']
        # ------------------------------------------------------------------------ #
        return self.new_obj, self.datas

    def create_obj(self, boxes, preds, faces):
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            pos = [(x2 + x1)//2, (y2 + y1)//2]
            self.new_obj.append({
                'faces': [faces[i]],
                'pos': pos,
                'id': np.random.randint(100),
                'name': self.get_true_names(preds[i]),
                'size': x2-x1,
                'pred': preds[i],
            })

    def update(self):
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
        if old is not None and new is not None:
            for name in self.obj[old]['pred']:
                if name in self.new_obj[new]['pred']:
                    self.new_obj[new]['pred'][name] += self.obj[old]['pred'][name]
                else:
                    self.new_obj[new]['pred'][name] = self.obj[old]['pred'][name]
            self.new_obj[new]['faces'].extend(self.obj[old]['faces'])
            self.new_obj[new]['id'] = self.obj[old]['id']
            self.new_obj[new]['name'] = self.get_true_names(self.new_obj[new]['pred'])
            self.obj[old].update(self.new_obj[new])
            self.obj[old]['dir'] = self.dir if self.obj[old]['size_0'] -\
                self.obj[old]['size'] < 0 else ''
            self.obj[old]['appear'] += 1
            self.obj[old]['disappear'] = 0
        elif old is not None:
            self.obj[old]['disappear'] += 1
            if self.obj[old]['disappear'] > self.max_disappear:
                if self.obj[old]['appear'] > self.min_appear:
                    self.export_obj(old)
        elif new is not None:
            self.obj.append(self.new_obj[new])
            self.obj[-1].update({
                'size_0': self.new_obj[new]['size'],
                'appear': 1,
                'disappear': 0
            })

    def export_obj(self, old):
        obj = self.obj.pop(old)
        if 'UNKOWN' == obj['name']:
            for i in range(len(obj['faces'])):
                with(open('log/unknown', 'w')) as file:
                    file.write(str(self.i))
                self.i += 1
                self.datas.append({
                    'timestamp': int(time.time()),
                    'camera': obj['dir'],
                    'name': '%s-%d' % (obj['name'], self.i),
                    'capture': image_encode(obj['faces'][i])
                })
        else:
            self.datas.append({
                'timestamp': int(time.time()),
                'camera': obj['dir'],
                'name': obj['name'],
                'capture': image_encode(obj['faces'][-3])
            })

    def get_true_names(self, preds):
        return preds
