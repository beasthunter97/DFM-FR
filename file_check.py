# region IMPORT
import argparse
import pickle
import time
import cv2
import numpy as np
from imutils.video import FPS

from lib.func import draw, face_detect, face_track, getNames
from lib.func import VideoCapture
# endregion


# region DEF
def config(d):
    f = open('config', 'r')
    data = f.read().split('\n')[0:2]
    data = list(map(lambda x: list(map(int, x.split(', '))), data))
    f.close()
    return data[d]


def transform(frame: np.array) -> np.array:
    try:
        frame = cv2.resize(frame, (width, height))
        boxes = face_detect(frame, crop_boundary)
        retval = [frame, boxes]
    except cv2.error:
        retval = None
    return retval


def boundary_update(crop_boundary: list, box: list) -> np.array:
    top, right, bottom, left = box
    if top - crop_boundary[0] <= 30:
        crop_boundary[0] -= 30 if crop_boundary[0] - \
            30 > 0 else crop_boundary[0]
    if crop_boundary[1] - right <= 50:
        crop_boundary[1] += 50 if crop_boundary[1] + \
            50 < width else width - crop_boundary[1]
    if crop_boundary[2] - bottom <= 30:
        crop_boundary[2] += 30 if crop_boundary[2] + \
            30 < height else height - crop_boundary[2]
    if left - crop_boundary[3] <= 30:
        crop_boundary[3] -= 30 if crop_boundary[3] - \
            30 > 0 else crop_boundary[3]
    return crop_boundary
# endregion


def main() -> None:
    # region CONST & INITIAL
    global crop_boundary
    crop_boundary = boundary_values.copy()
    video = VideoCapture(src, transform=transform).start()
    obj = []
    counter = 0
    writer = None
    fps = FPS().start()
    # endregion

    while True:

        frame_boxes = video.read()
        if frame_boxes is None:
            break

        frame, boxes = frame_boxes
        names = getNames(frame_boxes, data)
        obj_new = []

        for (i, box) in enumerate(boxes):
            obj_new.append({
                'pos': [(box[1] + box[3]) // 2, (box[2] + box[0]) // 2],
                'frame': frame,
                'size0': max(box[1] - box[3], box[2] - box[0]),
                'size': max(box[1] - box[3], box[2] - box[0]),
                'names': names[i],
                'direction': '',
                'true_name': '',
                'count': 0,
                'cont': 1,
                'box': i,
            })
            crop_boundary = boundary_update(crop_boundary, box)

        if len(obj) != 0:
            obj = face_track(obj, obj_new, args['direction'])
            counter = 0
        else:
            obj = obj_new
            counter += 1
            if counter >= 5:
                crop_boundary = boundary_values.copy()
        if boxes != []:
            info = [[i.get('true_name'), i.get('box')] for i in obj
                    if i.get('count') == 0]
            draw(frame, boxes, info)

        # region INFO2FRAME
        # cv2.rectangle(frame, (crop_boundary[3], crop_boundary[2]),
        #               (crop_boundary[1], crop_boundary[0]), (0, 255, 0), 2)
        text = '{}: {}'.format(args['direction'], inout[0])
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 2)
        frame = cv2.resize(frame, (720, 560))
        cv2.imshow("CAM", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        # endregion

        if writer is None and args['output'] is not None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args['output'], fourcc, 20,
                                     (frame.shape[1], frame.shape[0]), True)
        if writer is not None:
            writer.write(frame)
        fps.update()

    if writer is not None:
        writer.release()
    video.stop()
    fps.stop()
    print(fps.fps())
    print(fps.elapsed())
    cv2.destroyAllWindows()


if __name__ == '__main__':

    # region ARGUMENT PARSER
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', required=True,
                    help='Path to input source')
    ap.add_argument('-o', '--output', default=None,
                    help='Out put MP4 file')
    ap.add_argument('-d', '--direction', default=None,
                    help='If input is not "cam-in" or "cam-out", specify In/Out')
    ap.add_argument('-l', '--log-folder', default='log',
                    help='Path to log folder')
    ap.add_argument('-e', '--encodings', default='encodings.pickle',
                    help='Path to face encodings file')
    args = vars(ap.parse_args())

    data = pickle.loads(open(args['encodings'], 'rb').read())
    log_dir = np.empty((2,), dtype='<U10')
    log_dir[0] = args['log_folder']
    log_dir[1] = args['log_folder'].split('/')[-1]

    src = args['input']
    if src == 'cam-in':
        src = 'rtsp://192.168.200.79:554/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream' # noqa
        args['direction'] = 'In'
        time.sleep(30)
    elif src == 'cam-out':
        src = 'rtsp://192.168.200.80:555/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream' # noqa
        args['direction'] = 'Out'
        time.sleep(30)
    elif '192.168' in src:
        src = 'rtsp://' + src + '/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream' # noqa
        time.sleep(30)

    if args['direction'] is None:
        raise SyntaxError('If input is not "cam-in" or "cam-out", \
            specify [-d][--direction] as In or Out')
    else:
        args['direction'] = args['direction'].capitalize()
        if args['direction'] == 'Out':
            boundary_values = config(0)
        else:
            boundary_values = config(1)
    # endregion

    inout = np.array((0, 0), dtype='int')
    boundary_values = [boundary_values[1],
                       boundary_values[0] + boundary_values[2],
                       boundary_values[1] + boundary_values[3],
                       boundary_values[0]]
    width = 1024
    height = 1536 * width // 2048
    main()
