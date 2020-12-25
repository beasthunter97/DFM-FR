"""
This file contains `classes` and `methods` for ``tflite`` models.
Modyfied from google's ``PyCoral API``. See PyCoral API for more details.
https://github.com/google-coral/pycoral/blob/master/pycoral/adapters/common.py

`Note: there can be major differences due to version change.`
"""
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

EDGETPU_SHARED_LIB = 'libedgetpu.so.1'


def make_interpreter(model_file):
    """
    Make tflite interpreter.

    Args:
        model_file (str): Path to tflite model file.

    Returns:
        tflite.Interpreter: Allocated interpreter.
    """
    model_file, *device = model_file.split('@')
    interpreter = tflite.Interpreter(
        model_path=model_file,
        experimental_delegates=[
            tflite.load_delegate(EDGETPU_SHARED_LIB,
                                 {'device': device[0]} if device else {})
        ]
    )
    interpreter.allocate_tensors()
    return interpreter


def set_input(interpreter, image):
    """
    Set image as the tflte interpreter's input.

    Args:
        interpreter (tflite.Interpreter): Tflite interpreter
        image (np.ndarray): Input image with shape (W, H, C), color channel's order
                            depends on the model.
    """
    input_tensor(interpreter)[:, :] = image


def input_image_size(interpreter):
    """
    Check input image's size required by the interpreter

    Args:
        interpreter (tflite.Interpreter): Tflite interpreter.

    Returns:
        tuple: (w, h, c).
    """
    _, height, width, channels = interpreter.get_input_details()[0]['shape']
    return width, height, channels


def input_tensor(interpreter):
    """
    Returns input tensor view as numpy ``array`` of shape (height, width, channel).

    Args:
        interpreter (tflite.Interpreter): Tflite interpreter.

    Returns:
        np.ndarray: Numpy ``array`` of shape (h, w, c).
    """
    tensor_index = interpreter.get_input_details()[0]['index']
    return interpreter.tensor(tensor_index)()[0]


def output_tensor(interpreter, i, to_int=False):
    """
    Returns dequantized output tensor if quantized before.
    `Important change is made to avoid numerical error from google's original API.`

    Args:
        interpreter (tflite.Interpreter): Tflite interpreter.
        i (int): Index of the output desired.
        to_int (str, optional): If set to true, output will be converted to ``int`` type.
                                This is required if the model's output is quantized and
                                output + zero_point > 255. Defaults to ``True``.

    Returns:
        np.ndarray: dequantized output if quantized before.
    """
    output_details = interpreter.get_output_details()[i]
    output_data = np.squeeze(interpreter.tensor(output_details['index'])())
    if to_int:
        output_data = output_data.astype(int)
    if 'quantization' not in output_details:
        return output_data
    scale, zero_point = output_details['quantization']
    if scale == 0:
        return output_data - zero_point
    return scale * (output_data - zero_point)


def fix_box(x1, y1, x2, y2):
    """
    Fix the face detection's bounding box to a square bounding box.

    Args:
        x1 (int): Point x1.
        y1 (int): Point y1.
        x2 (int): Point x2.
        y2 (int): Point y2.

    Returns:
        tuple: square box (x1, y1, x2, y2)
    """
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
    """
    Face detector class.

    Simplify the PyCoral API detection model syntax.
    """
    def __init__(self, model_path, min_face_size, threshold=0.3, face_size=96):
        """
        Class initialize.

        Args:
            model_path (str): Path to tflite detection model.
            min_face_size (int): Minimum face size to detect.
            threshold (float, optional): Threshold. Defaults to ``0.3``.
            face_size (int, optional): Size of output face. Defaults to ``96``.
        """
        self.model = make_interpreter(model_path)
        self.min_face_size = min_face_size
        self.threshold = threshold
        if isinstance(face_size, int):
            self.face_size = (face_size, face_size)
        else:
            self.face_size = face_size

    def detect(self, image, return_faces=False):
        """
        Detect image.

        Args:
            image (np.ndarray): Original image.
            return_faces (bool, optional): If set to True, resized faces is returned.
                                           Defaults to ``False``.

        Returns:
            tuple or list: A list of bounding box [x1, y1, x2, y2]. If return_faces is\
                           True, return a tuple of bounding box list and face list.
        """
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
    """
    Face detector class.

    Simplify the PyCoral API classification model syntax.
    """
    def __init__(self, model_path, labels, top_k=3, threshold=0.5):
        """
        [summary]

        Args:
            model_path (str): Path to tflite face recognize model.
            labels (str): Path to label file.
            top_k (int, optional): Top k highest probability. Defaults to ``3``.
            threshold (float, optional): Threshold. Defaults to ``0.5``.
        """
        if model_path is None or labels is None:
            self.model = None
        else:
            with open(labels, 'r') as file:
                self.labels = file.read().split('\n')
            self.model = make_interpreter(model_path)
            self.top_k = top_k
            self.threshold = threshold

    def recognize(self, images):
        """
        Recognize face from images.

        Args:
            images (list): List of input images to recoginize.

        Returns:
            list: List of dictionaries with maximum of `k` items ``name: prob``.\
                  Number of dictionaries equal to number of input images
        """
        if self.model is not None:
            names = []
            for image in images:
                set_input(self.model, image)
                self.model.invoke()
                # ----------------------------------------------------------------
                output = output_tensor(self.model, 0, job='classify')
                index = np.argpartition(output, -3)[-3:]
                name = {}
                for i in index:
                    if output[i] < self.threshold:
                        continue
                    name.update({self.labels[i]: output[i]})
                if name == {}:
                    name = {'UNKNOWN': 1}
                names.append(name)
                # ----------------------------------------------------------------
        else:
            names = [{'UNKNOWN': 1}] * len(images)
        return names
