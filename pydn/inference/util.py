import json
import os
import pickle

import threading
from threading import Thread
import cv2
import numpy as np
import requests
from threading import Thread, Lock
from multiprocessing import Queue

from pydn.inference.models import DNChip, ChipMeta
from pydn.inference.nms import non_max_suppression_fast


def processImage(chip_metas):
    chips = []
    for chip_meta in chip_metas:
        img_fn = chip_meta[0]
        json_res = chip_meta[1]
        im = cv2.imread(img_fn)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        for chip in json_res:
            cm = ChipMeta(image_fn=img_fn, chip_meta=chip)
            crop = im[chip['y1']:chip['y2'], chip['x1']:chip['x2']]
            custom_crop = cv2.resize(
                crop, (832, 832), interpolation=cv2.INTER_NEAREST)
            custom_crop = custom_crop.transpose(2, 0, 1)

            dnc = DNChip(chip_meta=cm, chip_im=custom_crop)
            chips.append(dnc)
    return chips


def draw_batch_detections(batch_detections, save_image_path):
    for batch_img_fn, batch_img_detections in batch_detections.items():
        boxes = []
        for det in batch_img_detections:
            boxes.append(np.array(det.box))
        boxes_pre = np.array(boxes)
        pick = non_max_suppression_fast(boxes_pre, .45)
        det_list = np.array(batch_img_detections)
        new_det_list = det_list[pick]
        removed_det_list = np.delete(det_list, pick)
        batch_im = cv2.imread(batch_img_fn)
        batch_im = cv2.cvtColor(batch_im, cv2.COLOR_BGR2RGB)
        # DRAW
        for det in new_det_list:
            boxColor = (0, 255, 0)
            cv2.rectangle(batch_im, (det.box[0], det.box[1] + 2),
                          (det.box[2], det.box[3] + 2), boxColor, 2)
        for det in removed_det_list:
            boxColor = (255, 0, 0)
            cv2.rectangle(batch_im, (det.box[0], det.box[1]),
                          (det.box[2], det.box[3]), boxColor, 2)
        out_path = os.path.join(save_image_path, os.path.basename(batch_img_fn))
        cv2.imwrite(out_path, batch_im)

# query api to get resulting chip dimensions
def get_chip_dims_api(chip_dim, img_base, host_fmt = "http://127.0.0.1:5000/api/image_chips?w={0}&h={0}&image_name={1}"):
    get_uri = host_fmt.format(chip_dim, img_base)
    r = requests.get(url=get_uri)
    if r.status_code != 200:
        return None
    json_res = json.loads(r.text)
    return json_res

def get_image_info_api(img_base, host_fmt = "http://127.0.0.1:5000/api/image?image_name={0}"):
    get_uri = host_fmt.format(img_base)
    r = requests.get(url=get_uri)
    if r.status_code != 200:
        return None
    json_res = json.loads(r.text)
    return json_res

def get_matches_api(boxes, img_base, host_fmt="http://127.0.0.1:5000/api/ml/nms?image_name={0}"):
    get_uri = host_fmt.format(img_base, img_base)
    headers = {'Content-type': 'application/json'}
    r = requests.post(url=get_uri, data=json.dumps(boxes), headers=headers)
    if r.status_code != 200:
        return None
    json_res = json.loads(r.text)
    return json_res


def save_dets(file, dets):
    with open(file, 'wb') as handle:
        pickle.dump(dets, handle)

def load(pkl_file):
    r = None
    if os.path.isfile(pkl_file):
        with open(pkl_file, "rb") as f:
            r = pickle.load(f, encoding='latin1')
        return r
    return r


import matplotlib.pyplot as plt
import numpy as np


def show_images(images, cols=1, titles=None, size_mul = 1):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()

    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        a.axis('off')
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    print(fig.get_size_inches())
    fig.set_size_inches(np.array(fig.get_size_inches()*size_mul) * n_images)
    plt.show()


def filter_confidence(detections, confidence_thresh):
    print("Confidence %.2f" % (confidence_thresh))
    new_dets = {}
    for im in detections:
        dets = detections[im]
        for det in dets:
            if det.confidence > confidence_thresh:
                if not im in new_dets:
                    new_dets[im] = []
                new_dets[im].append(det)

    return new_dets

def normalize_thermal(thermal_image, percent=0.01):

    if not thermal_image is None and thermal_image.dtype is not np.dtype('uint8'):
        thermal_norm = np.floor(( thermal_image -
              np.percentile( thermal_image, percent)) /
              (np.percentile( thermal_image, 100 - percent) -
              np.percentile( thermal_image, percent )) * 256 )
    else:
        thermal_norm = thermal_image

    return thermal_norm.astype( np.uint8 )