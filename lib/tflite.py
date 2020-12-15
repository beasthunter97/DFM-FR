import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

EDGETPU_SHARED_LIB = 'libedgetpu.so.1'


def make_interpreter(model_file):
    model_file, *device = model_file.split('@')
    return tflite.Interpreter(
        model_path=model_file,
        experimental_delegates=[
            tflite.load_delegate(EDGETPU_SHARED_LIB,
                                 {'device': device[0]} if device else {})
        ]
    )


def set_input(interpreter, image):
    """Copies data to input tensor."""
    input_tensor(interpreter)[:, :] = image


def input_image_size(interpreter):
    """Returns input image size as (width, height, channels) tuple."""
    _, height, width, channels = interpreter.get_input_details()[0]['shape']
    return width, height, channels


def input_tensor(interpreter):
    """Returns input tensor view as numpy array of shape (height, width, 3)."""
    tensor_index = interpreter.get_input_details()[0]['index']
    return interpreter.tensor(tensor_index)()[0]


def output_tensor(interpreter, i):
    """Returns dequantized output tensor if quantized before."""
    output_details = interpreter.get_output_details()[i]
    output_data = np.squeeze(interpreter.tensor(output_details['index'])())
    if 'quantization' not in output_details:
        return output_data
    scale, zero_point = output_details['quantization']
    if scale == 0:
        return output_data - zero_point
    return scale * (output_data - zero_point)


def fix_box(x1, y1, x2, y2):
    w = x2 - x1
    h = y2 - y1
    if h < w:
        d = (w - h) // 2
        x1 += d
        x2 -= d + w - h - d*2
    elif w < h:
        d = (h - w) // 2
        y1 += d
        y2 -= d + h - w - d*2
    return x1, y1, x2, y2


class Detector:
    def __init__(self, model_path, min_face_size=0, threshold=0.3, face_size=112):
        self.model = make_interpreter(model_path)
        self.model.allocate_tensors()
        self.min_face_size = min_face_size
        self.threshold = threshold
        if isinstance(face_size, int):
            self.face_size = (face_size, face_size)
        else:
            self.face_size = face_size

    def detect(self, image, return_faces=False):
        h, w = image.shape[:2]
        inp = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inp = cv2.resize(inp, (320, 320))
        set_input(self.model, inp)
        self.model.invoke()
        confidence = output_tensor(self.model, 2)
        output_boxes = output_tensor(self.model, 0)
        boxes = []
        for i in range(int(output_tensor(self.model, 3))):
            if confidence[i] > self.threshold:
                y1, x1, y2, x2 = output_boxes[i]
                x1, x2 = max(0, int(w * x1)), min(w, int(x2 * w))
                y1, y2 = max(0, int(h * y1)), min(h, int(y2 * h))
                x1, y1, x2, y2 = fix_box(x1, y1, x2, y2)
                if self.min_face_size <= (y2 - y1):
                    boxes.append([x1, y1, x2, y2])
        if return_faces:
            faces = []
            for x1, y1, x2, y2 in boxes:
                face = image[y1:y2, x1:x2].copy()
                faces.append(cv2.resize(face, self.face_size))
            return boxes, faces
        else:
            return boxes


class Recognizer:
    def __init__(self, model_path, labels):
        if model_path is None or labels is None:
            self.model = None
        else:
            with open(labels, 'r') as file:
                self.labels = file.read().split('\n')
            self.model = make_interpreter(model_path)
            self.model.allocate_tensors()

    def recognize(self, images):
        if self.model is not None:
            names = []
            for image in images:
                #set_input(self.model, (image-127.5)/128)
                set_input(self.model, image)
                self.model.invoke()
                # ----------------------------------------------------------------
                output = output_tensor(self.model, 0)
                index = np.argpartition(output, -1)[-1:]
                print(output)
                name = {}
                for i in index:
                    # if output_tensor(self.model, 0)[i] < 0.9:
                    #     continue
                    # if i not in [2, 9, 10, 11]:
                    #     continue
                    name.update({self.labels[i]: output[i]})
                if name == {}:
                    name['UNKNOWN'] = 1
                names.append(name)
                # ----------------------------------------------------------------
        else:
            names = [{'UNKNOWN': 1}] * len(images)
        return names
