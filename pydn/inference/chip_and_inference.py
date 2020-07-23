import multiprocessing

import pickle
import time

from pydn.inference.darknet import MultiGPUBatchDetectors
from pydn.inference.util import *
from pydn.inference.models import Detection

dn_config_fn = "/fast/experiments/yolov4_3c_832/base/yolov4-custom.cfg"
dn_meta_fn = "/fast/experiments/yolov4_3c_832/base/yolo.data"
dn_weight_fn = "/fast/experiments/yolov4_3c_832/base/weights/yolov4-custom_best.weights"
img_list_fn = "/data2/2019/all_rgb_images.txt"

img_list = open(img_list_fn).read().split('\n')
img_list = list(line for line in img_list if line and os.path.splitext(line)[1] != ".tif") #remove possible empty lines

DRAW_DETS = False
SAVE_IMAGE_OUT = '/data2/test/'
DETECTIONS_PKL_LOC = '/fast/experiments/yolov4_3c_832/base/inference/kotz/kotz_fl01_fl04_fl05_detections_2.pkl'
if DRAW_DETS:
    print("DRAWING BOXES AND SAVING IMAGES IN %s" % SAVE_IMAGE_OUT)

CHIP_DIMENSION = 832
MIN_CHIP_BUFFER = 1
START_IDX = 37400



# I think we want an odd batch size number because with an even we risk never being divisible by total chips if they for some reason were to be odd
mGPUBatchDetector = MultiGPUBatchDetectors(dn_config_fn, dn_meta_fn, dn_weight_fn, dim=CHIP_DIMENSION, gpus=[1], batch_sizes=[1],
                                           nms_thresh=.45, hier_thresh=.5, thresh=.1)

pool = multiprocessing.Pool(processes=2)

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
last_saved = START_IDX
for img_fn in img_list[START_IDX:]:
    num_images += 1
    if img_fn in detections and len(detections[img_fn]) > 0:
        total_detections += len(detections[img_fn])
        continue

    total_time_start = time.time() # reset total batch time

    img_base = os.path.basename(img_fn)
    batch_detections[img_fn] = []
    json_res = get_chip_dims_api(CHIP_DIMENSION, img_base)
    if json_res is None:
        continue

    chip_meta_queue[img_fn] = json_res
    chip_meta_queue_num_chips += len(json_res)

    if chip_meta_queue_num_chips >= MIN_CHIP_BUFFER and chip_meta_queue_num_chips % mGPUBatchDetector.total_batch_size == 0:
        load_chip_time_total_start = time.time()
        items = list(chip_meta_queue.items())
        n=5
        items = [items[i:i + n] for i in range(0, len(items), n)]
        results = pool.map(processImage, items)
        for r in results:
            chip_queue = chip_queue + r
        load_chip_time_total += time.time() - load_chip_time_total_start
    if len(chip_queue) >= MIN_CHIP_BUFFER and len(chip_queue) % mGPUBatchDetector.total_batch_size == 0:
        print("Starting Batch Inference %d chips, %d images" % (len(chip_queue), len(batch_detections.items())))
        inference_time_start = time.time()
        batch_results = mGPUBatchDetector.detect(chip_queue)
        assert (len(mGPUBatchDetector.image_queue) == 0) # assert detector got through all images
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

        # save detections
        if total_batch_detections > 0 and num_images-last_saved >= 500:
            last_saved=num_images
            print("Saving to %s" % DETECTIONS_PKL_LOC)
            save_dets(DETECTIONS_PKL_LOC, detections)

        inference_time_total = 0 # reset inference time total
        load_chip_time_total = 0 # reset chipping time total

        chip_queue = [] # reset chip queue
        chip_meta_queue = {}
        chip_meta_queue_num_chips = 0
        batch_detections = {} # reset batch detection info


pool.close()

save_dets(DETECTIONS_PKL_LOC,detections)
print("Finished")