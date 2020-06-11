import multiprocessing

import cPickle as pickle
import time

import cv2
import json
import os
import numpy as np
import requests

from pydn.inference.ImageProcessor import processImage, draw_batch_detections
from pydn.inference.darknet import performBatchDetect, load_net_meta_batch
from pydn.inference.models import ChipMeta, DNChip, Detector, Detection, MultiGPUBatchDetectors
from pydn.inference.nms import non_max_suppression_fast

dn_config_fn = "/fast/experiments/yolov4_3c_832/base/yolov4-custom.cfg"
dn_meta_fn = "/fast/experiments/yolov4_3c_832/base/yolo.data"
dn_weight_fn = "/fast/experiments/yolov4_3c_832/base/weights/yolov4-custom_best.weights"
detector = Detector(dn_config_fn, dn_meta_fn, dn_weight_fn)
img_list_fn = "/data2/2019/all_rgb_images.txt"

img_list = open(img_list_fn).read().split('\n')
img_list = list(line for line in img_list if line and os.path.splitext(line)[1] != ".tif") #remove possible empty lines

host = "http://127.0.0.1:5000/api/image_chips?w={0}&h={0}&image_name={1}"

DRAW_DETS = False
SAVE_IMAGE_OUT = '/data2/test/'
DETECTIONS_PKL_LOC = '/fast/experiments/yolov4_3c_832/base/inference/kotz/kotz_fl01_fl04_fl05_detections.pkl'

CHIP_DIMENSION = 832
MIN_CHIP_BUFFER = 500
START_IDX = 17300

def save_dets(dets):
    with open(DETECTIONS_PKL_LOC, 'wb') as handle:
        pickle.dump(dets, handle)

# I think we want an odd batch size number because with an even we risk never being divisible by total chips if they
# for some reason were to be odd
mGPUBatchDetector = MultiGPUBatchDetectors(dn_config_fn, dn_meta_fn, dn_weight_fn, dim=CHIP_DIMENSION, gpus=[0,1], batch_sizes=[3,3],
                                           nms_thresh=.45, hier_thresh=.5, thresh=.1)

pool = multiprocessing.Pool(processes=8)

chip_queue = []
# load existing detections if already exist
detections = {}
if os.path.isfile(DETECTIONS_PKL_LOC):
    with open(DETECTIONS_PKL_LOC, "rb") as f:
        detections = pickle.load(f)
num_images = START_IDX
total_images = len(img_list)
total_detections = sum([len(value) for key, value in detections.items()])
batch_detections = {}
total_time_start = 0
inference_time_total = 0
load_chip_time_total = 0
chip_meta_queue = {}
chip_meta_queue_num_chips = 0
for img_fn in img_list[START_IDX:]:
    num_images += 1
    if img_fn in detections and len(detections[img_fn]) > 0:
        total_detections += len(detections[img_fn])
        continue

    total_time_start = time.time() # reset total batch time

    img_base = os.path.basename(img_fn)
    batch_detections[img_fn] = []
    get_uri = host.format(CHIP_DIMENSION, img_base)
    r = requests.get(url=get_uri)
    if r.status_code != 200:
        continue
    json_res = json.loads(r.text)
    chip_meta_queue[img_fn] = json_res
    chip_meta_queue_num_chips += len(json_res)

    if chip_meta_queue_num_chips > MIN_CHIP_BUFFER and chip_meta_queue_num_chips % mGPUBatchDetector.total_batch_size == 0:
        load_chip_time_total_start = time.time()
        items = list(chip_meta_queue.items())
        results = pool.map(processImage, items)
        for r in results:
            chip_queue = chip_queue + r
        load_chip_time_total += time.time() - load_chip_time_total_start
    if len(chip_queue) > MIN_CHIP_BUFFER and len(chip_queue) % mGPUBatchDetector.total_batch_size == 0:
        print("Starting Batch Inference %d chips, %d images" % (len(chip_queue), len(batch_detections.items())))
        inference_time_start = time.time()
        batch_results = mGPUBatchDetector.detect(chip_queue)
        inference_time_only = (time.time() - inference_time_start)
        inference_time_total += inference_time_only
        for br in batch_results:
            batch_boxes = br.batch_boxes
            batch_scores = br.batch_scores
            batch_classes = br.batch_classes
            batch_meta = br.meta_batch
            if np.any(batch_boxes) > 0:
                for i in range(br.nbatch):
                    if np.any(batch_boxes[i]):
                        boxes = np.array(batch_boxes[i])
                        scores = np.array(batch_scores[i])
                        classes = np.array(batch_classes[i])
                        meta = batch_meta[i].chip_meta
                        file = batch_meta[i].image_fn
                        boxes[:, 0] += meta['x1']
                        boxes[:, 1] += meta['y1']
                        boxes[:, 2] += meta['x1']
                        boxes[:, 3] += meta['y1']

                        if not file in detections:
                            detections[file] = []
                        for j, box in enumerate(boxes):
                            det = Detection(box, classes[j], scores[j])
                            detections[file].append(det)
                            batch_detections[file].append(det)
                            total_detections+=1


        if DRAW_DETS:
            draw_batch_detections(batch_detections, SAVE_IMAGE_OUT)

        for batch_img_fn, batch_img_detections in batch_detections.items():
            batch_img_fn_base = os.path.basename(batch_img_fn)
            print("%s - %d detections" % (batch_img_fn_base,len(batch_img_detections)))
        total_time = time.time() - total_time_start
        total_time_per_im = total_time/len(batch_detections.items())
        total_batch_detections = sum([len(value) for key, value in batch_detections.items()])
        print("total time %s, per image %s" % (total_time, total_time_per_im))
        print("inference total time %s, per image %s" % (inference_time_total, inference_time_total/len(batch_detections.items())))
        print("loading/chipping total time %s, per image %s" % (load_chip_time_total, load_chip_time_total/len(batch_detections.items())))
        print("%d/%d images complete - %d batch detections - %d total detections\n" % (num_images, total_images, total_batch_detections, total_detections))
        assert (len(mGPUBatchDetector.image_queue) == 0) # assert detector got through all images

        # save detections
        if total_batch_detections > 0:
            print("Saving to %s" % DETECTIONS_PKL_LOC)
            save_dets(detections)

        inference_time_total = 0 # reset inference time total
        load_chip_time_total = 0 # reset chipping time total

        chip_queue = [] # reset chip queue
        chip_meta_queue = {}
        chip_meta_queue_num_chips = 0
        batch_detections = {} # reset batch detection info


pool.close()

save_dets(detections)
print("Finished")