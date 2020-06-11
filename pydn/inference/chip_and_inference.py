import time

import cv2
import json
import os
import numpy as np
import requests
from darknet import performBatchDetect, load_net_meta_batch

class ChipMeta():
    def __init__(self, image_fn, chip_meta):
        self.image_fn = image_fn
        self.chip_meta = chip_meta

class DNChip():
    def __init__(self, chip_meta, chip_im):
        self.chip_meta = chip_meta
        self.chip_im = chip_im

class Detection():
    def __init__(self, box, classid, confidence):
        self.box = box
        self.classid = classid
        self.confidence = confidence

dn_config_fn = "/fast/experiments/yolov4_3c_832/base/yolov4-custom.cfg"
dn_meta_fn = "/fast/experiments/yolov4_3c_832/base/yolo.data"
dn_weight_fn = "/fast/experiments/yolov4_3c_832/base/weights/yolov4-custom_best.weights"

img_list_fn = "/data2/2019/all_rgb_images.txt"
dim = 832

img_list = open(img_list_fn).read().split('\n')
img_list = list(line for line in img_list if line)  #remove possible empty lines

host = "http://127.0.0.1:5000/api/image_chips?w={0}&h={0}&image_name={1}"
chip_dim = 832
batch_size = 5

net, net_meta = load_net_meta_batch(configPath=dn_config_fn, weightPath=dn_weight_fn, metaPath=dn_meta_fn, batchSize=batch_size)

boxes = []
scores = []
classes = []
chip_meta = []
chip_queue = []
detections = {}
for img_fn in img_list:
    start_time = time.time()
    img_base = os.path.basename(img_fn)
    get_uri = host.format(chip_dim, img_base)
    r = requests.get(url=get_uri)
    if r.status_code != 200:
        print("ERROR %s\n%s" % (img_base, r.text))
        continue
    json_res = json.loads(r.text)
    im = cv2.imread(img_fn)
    for chip in json_res:
        cm = ChipMeta(image_fn=img_fn, chip_meta=chip)
        crop = im[chip['y1']:chip['y2'], chip['x1']:chip['x2']]
        dnc = DNChip(chip_meta=cm, chip_im = crop)
        chip_queue.append(dnc)

    while(len(chip_queue) >= batch_size):
        # make batch
        batch_meta = []
        batch_chips = []
        for i in range(batch_size):
            dn_chip = chip_queue.pop()
            batch_meta.append(dn_chip.chip_meta)
            batch_chips.append(dn_chip.chip_im)

        batch_boxes, batch_scores, batch_classes = performBatchDetect(net, net_meta, image_list=batch_chips, pred_height=832, pred_width=832, c=3, batch_size=batch_size)
        if np.any(batch_boxes) > 0:
            for i in range(batch_size):
                if np.any(batch_boxes[i]):
                    boxes = np.array(batch_boxes[i])
                    scores = np.array(batch_scores[i])
                    classes = np.array(batch_classes[i])
                    meta = batch_meta[i]
                    im_fn = batch_meta[i].image_fn
                    meta = batch_meta[i].chip_meta
                    boxes[:, 0] += meta['x1']
                    boxes[:, 1] += meta['y1']
                    boxes[:, 2] += meta['x1']
                    boxes[:, 3] += meta['y1']
                    if not im in detections:
                        detections[im] = []

                    detections[im].append()

        x=1
    print("--- %s seconds ---" % (time.time() - start_time))
    x=1
