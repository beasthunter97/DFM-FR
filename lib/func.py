import os
import time

import cv2
import face_recognition
import numpy as np
import pandas as pd
from mtcnn import MTCNN
from scipy.spatial import distance

import __main__

detector = MTCNN(min_face_size=110, steps_threshold=[0.6, 0.8, 0.85])
no_frame = 7
min_frame = 3


def face_detect(img: np.array, crop_boundary: np.array) -> list:

    _t, _r, _b, _l = crop_boundary
    boxes = []
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img[_t:_b, _l:_r])

    for face in faces:
        x, y, w, h = face['box']
        x += _l
        y += _t
        boxes.append([y, x + w, y + h, x])
    return boxes


def draw(frame: np.array, boxes: list, info: list) -> None:

    if boxes != []:
        for _info in info:
            text = _info[0]
            try:
                top, right, bottom, left = boxes[_info[1]]
            except IndexError:
                print(boxes, info, _info)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            name_pos = top - 15 if top - 15 > 45 else bottom + 40
            cv2.putText(frame, text, (left, name_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)


def getNames(frame_boxes: list, data: dict, tol1=0.45, tol2=0.1) -> dict:

    names_all = []
    encodings = face_recognition.face_encodings(frame_boxes[0], frame_boxes[1])

    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding, tol1) # noqa
        names = {}

        if True in matches:
            matchesIndxs = [i for (i, b) in enumerate(matches) if b]

            for i in matchesIndxs:
                name = data["names"][i]
                totalname = data["names"].count(name)
                names[name] = names.get(name, 0) + 1 / totalname

            _names = names.copy()
            for i in _names:
                if names.get(i) < tol2:
                    names.pop(i)
        names_all.append(names)
    return names_all


def write_log(frame: np.array, name: str, direction: str) -> None:

    log_dir = __main__.log_dir

    [date, hour] = time.strftime('%d/%m/%Y,%H:%M:%S').split(',')
    path = '/'.join([log_dir[0], name,
                     date.replace('/', '.')])
    os.makedirs(path, exist_ok=True)
    path += '/' + hour + direction + '.jpg'
    cv2.imwrite(path, frame)
    timedata = pd.DataFrame({"Date": [date],
                             "Time": [hour],
                             "In/Out": [direction],
                             "Name": [name.strip()],
                             "Path": [path]})
    timedata.to_csv(log_dir[1] + '/log.csv', sep='\t',
                    mode='a', index=False, header=False)


def face_track(obj: list, obj_new: list, direction: str) -> list:

    if obj_new != []:
        pos = np.array([x.get('pos') for x in obj])
        pos_new = np.array([x.get('pos') for x in obj_new])
        D = distance.cdist(pos, pos_new)
        same = min(len(D), len(D[0]))
        usedRows = set()
        usedCols = set()

        for _ in range(same):
            col = D.min(axis=0).argsort()[0]
            row = D.min(axis=1).argsort()[0]
            if D[row, col] > obj[row]['size']:
                break
            # if not same_face_condition(obj[row], obj_new[col]):
            #     D[row, col] = 1000
            #     continue
            usedCols.add(col)
            usedRows.add(row)
            D[row, :] = 1000
            D[:, col] = 1000
            obj[row] = obj_update(obj[row], obj_new[col])

        obj = direction_check(obj, direction, usedRows)
        for i in range(len(obj_new)):
            if i in usedCols:
                continue
            obj.append(obj_new[i])

    else:
        obj = direction_check(obj, direction)
    return obj


def obj_update(obj: list, obj_new: list) -> list:

    for i in obj['names']:
        if i in obj_new['names']:
            obj['names'][i] = obj_new['names'].get(i) + obj['names'].get(i) # noqa
    obj_new['names'].update(obj['names'])

    size = obj['size0']
    cont = obj['cont'] + 1
    obj.update(obj_new)
    obj['size0'] = size
    obj['cont'] = cont
    obj['true_name'] = true_name(obj['names'])
    return obj


def direction_check(obj: np.array, direction: str, usedRows=[]):

    inout = __main__.inout

    for i in range(len(obj) - 1, -1, -1):
        if i in usedRows:
            continue
        if obj[i]['count'] >= no_frame:
            obj.remove(obj[i])
        else:
            obj[i]['count'] += 1
            if obj[i]['count'] == no_frame:
                if obj[i]['cont'] < min_frame:
                    continue
                if obj[i]['size'] > obj[i]['size0']:
                    write_log(obj[i]['frame'], obj[i]['true_name'], direction)
                    inout[-1] = 1
                    inout[0] += 1
                # elif obj[i]['size'] < obj[i]['size0']:
                #     write_log(obj[i]['frame'], obj[i]['true_name'], direction)
                #     inout[-1] = 1
                #     inout[int(not out)] += 1
    return obj


def true_name(names: dict) -> str:

    try:
        name = max(names, key=names.get)
    except ValueError:
        name = 'UNKNOWN'
    return name


def same_face_condition(old: dict, new: dict) -> bool:

    relative_size = abs(old['size'] - new['size']) / old['size']

    if relative_size > 0.2:
        same = False
    elif abs(old['pos'][1] - new['pos'][1]) > old['size'] // 5:
        same = False
    else:
        same = True
    return same
