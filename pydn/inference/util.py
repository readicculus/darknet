import json
import os
import cPickle as pickle

import threading
from threading import Thread
import cv2
import requests
from threading import Thread, Lock
from multiprocessing import Queue

from pydn.inference.models import DNChip, ChipMeta, np
from pydn.inference.nms import non_max_suppression_fast


def processImage(chip_meta):
    img_fn = chip_meta[0]
    json_res = chip_meta[1]
    im = cv2.imread(img_fn)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    chips = []
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

def save_dets(file, dets):
    with open(file, 'wb') as handle:
        pickle.dump(dets, handle)