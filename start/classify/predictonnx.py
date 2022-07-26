import numpy as np    # we're going to use numpy to process input and output data
import onnxruntime    # to inference ONNX models, we use the ONNX Runtime
from urllib.request import urlopen
import json
import time
import torchvision.transforms as transforms
import logging
import os
import sys
from datetime import datetime

# display images in notebook
from PIL import Image
import cv2


def load_labels(path):
    with open(path) as f:
        data = json.load(f)
    return np.asarray(data)


class ImageNotOpenableError(Exception):
    pass


# Run the model on the backend
d = os.path.dirname(os.path.abspath(__file__))
modelfile = os.path.join(d, 'detect_emotion.onnx')
labelfile = os.path.join(d, 'labels.json')

session = onnxruntime.InferenceSession(modelfile, None)

# get the name of the first input of the model
input_name = session.get_inputs()[0].name

labels = load_labels(labelfile)


def preprocess(input_data):
    img_data = Image.fromarray(resize_image_to_square(input_data))

    mu, st = 0, 255

    pre_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.CenterCrop(40),
        transforms.ToTensor(),
        transforms.Normalize(mean=(mu,), std=(st,))
    ])

    transformed_data = pre_transforms(img_data)
    transformed_data = transformed_data.unsqueeze(0)

    return transformed_data.numpy()


def softmax(x):
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def postprocess(result):
    return softmax(np.array(result)).tolist()


def resize_image_to_square(im: np.ndarray, desired_size=40):
    im = Image.fromarray(im)
    old_size = im.size

    ratio = float(desired_size)/min(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    im = im.resize(new_size, Image.Resampling.LANCZOS)
    im_arr = np.array(im)

    new_arr_size = im_arr.shape
    x_diff = (new_arr_size[0] - desired_size) // 2
    y_diff = (new_arr_size[1] - desired_size) // 2

    new_im_arr = im_arr[x_diff:new_arr_size[0]-x_diff, y_diff:new_arr_size[1]-y_diff]

    return new_im_arr


def crop_faces(im: Image.Image, padding=8):
    im = im.convert("L")
    im_arr = np.array(im)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(
        im_arr, scaleFactor=1.25, minNeighbors=3, minSize=(40, 40))

    print('Number of faces detected:', len(faces))

    crops = []

    for count, face in enumerate(faces):
        x, y, w, h = face

        left = x - padding
        right = x + w + padding
        top = y - padding
        bottom = y + h + padding
        crop = im_arr[left:right, top:bottom]

        crops.append({'img': crop, 'weight': x+y})

    # ensure that faces are sorted in the order in which they appear
    crops.sort(key=lambda x: x['weight'])
    crops = [c['img'] for c in crops]

    return faces, crops


def run_single_inference(raw_input):
    input_data = preprocess(raw_input)

    raw_result = session.run([], {input_name: input_data})

    res = postprocess(raw_result)

    idx = np.argmax(res)

    label = labels[idx]
    conf = np.round(res[idx], 2)

    results_rounded = [np.round(n, 2) for n in res]

    return label, conf, results_rounded


def predict(image: Image.Image):
    faces, crops = crop_faces(image)

    results = []

    start = time.time()
    for crop in crops:
        singe_result = run_single_inference(crop)
        results.append(singe_result)

    end = time.time()
    inference_time = np.round((end - start) * 1000, 2)

    return results, inference_time, faces.tolist()


def predict_image_from_file(file):
    with Image.open(file) as image:
        # imnew = ImageOps.fit(image, (224, 224))
        results, inference_time, faces = predict(image)

        response = {
            'created': datetime.utcnow().isoformat(),
            'results': results,
            'latency': inference_time,
            'faces': faces
        }
        logging.info(f'returning {response}')
        return response


def predict_image_from_url(image_url):
    try:
        with urlopen(image_url) as testImage:
            image = Image.open(testImage)
    except:
        logging.error("Unable to open URL")
        raise ImageNotOpenableError("Unable to open URL")

    # imnew = ImageOps.fit(image, (224, 224))

    results, inference_time, faces = predict(image)

    response = {
        'created': datetime.utcnow().isoformat(),
        'results': results,
        'latency': inference_time,
        'faces': faces
    }
    logging.info(f'returning {response}')
    return response


# if __name__ == '__main__':
#     print(predict_image_from_url(sys.argv[1]))

if __name__ == "__main__":
    img = np.array(Image.open("../sample.jpeg"))
    # new_img = resize_image_to_square(img)

    # preprocessed_data = preprocess(new_img)
    results = run_single_inference(img)

    print(results)
    
