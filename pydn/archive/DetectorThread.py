#!python3
"""
Python 3 wrapper for identifying objects in images

Requires DLL compilation

Both the GPU and no-GPU version should be compiled; the no-GPU version should be renamed "yolo_cpp_dll_nogpu.dll".

On a GPU system, you can force CPU evaluation by any of:

- Set global variable DARKNET_FORCE_CPU to True
- Set environment variable CUDA_VISIBLE_DEVICES to -1
- Set environment variable "FORCE_CPU" to "true"


To use, either run performDetect() after import, or modify the end of this file.

See the docstring of performDetect() for parameters.

Directly viewing or returning bounding-boxed images requires scikit-image to be installed (`pip install scikit-image`)


Original *nix 2.7: https://github.com/pjreddie/darknet/blob/0f110834f4e18b30d5f101bf8f1724c34b7b83db/python/darknet.py
Windows Python 2.7 version: https://github.com/AlexeyAB/darknet/blob/fc496d52bf22a0bb257300d3c79be9cd80e722cb/build/darknet/x64/darknet.py

@author: Philip Kahn
@date: 20180503
"""
#pylint: disable=R, W0401, W0614, W0703
import cv2
from ctypes import *
import math
import random
import os

import numpy as np


def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int),
                ("uc", POINTER(c_float)),
                ("points", c_int)]

class DETNUMPAIR(Structure):
    _fields_ = [("num", c_int),
                ("dets", POINTER(DETECTION))]

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]



class Detector(object):
    def __init__(self):
        self.lib = CDLL("/home/yuval/Documents/XNOR/sealnet/models/darknet/libdarknet.so", RTLD_LOCAL)
        self.hasGPU = True
        if self.hasGPU:
            self.set_gpu = self.lib.cuda_set_device
            self.set_gpu.argtypes = [c_int]


        self.lib.network_width.argtypes = [c_void_p]
        self.lib.network_width.restype = c_int
        self.lib.network_height.argtypes = [c_void_p]
        self.lib.network_height.restype = c_int

        self.copy_image_from_bytes = self.lib.copy_image_from_bytes
        self.copy_image_from_bytes.argtypes = [IMAGE, c_char_p]



        self.predict = self.lib.network_predict_ptr
        self.predict.argtypes = [c_void_p, POINTER(c_float)]
        self.predict.restype = POINTER(c_float)

        if self.hasGPU:
            self.set_gpu = self.lib.cuda_set_device
            self.set_gpu.argtypes = [c_int]

        self.make_image = self.lib.make_image
        self.make_image.argtypes = [c_int, c_int, c_int]
        self.make_image.restype = IMAGE

        self.get_network_boxes = self.lib.get_network_boxes
        self.get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int),
                                      c_int]
        self.get_network_boxes.restype = POINTER(DETECTION)

        self.make_network_boxes = self.lib.make_network_boxes
        self.make_network_boxes.argtypes = [c_void_p]
        self.make_network_boxes.restype = POINTER(DETECTION)

        self.free_detections = self.lib.free_detections
        self.free_detections.argtypes = [POINTER(DETECTION), c_int]

        self.free_batch_detections = self.lib.free_batch_detections
        self.free_batch_detections.argtypes = [POINTER(DETNUMPAIR), c_int]

        self.free_ptrs = self.lib.free_ptrs
        self.free_ptrs.argtypes = [POINTER(c_void_p), c_int]

        self.network_predict = self.lib.network_predict_ptr
        self.network_predict.argtypes = [c_void_p, POINTER(c_float)]

        self.reset_rnn = self.lib.reset_rnn
        self.reset_rnn.argtypes = [c_void_p]

        self.load_net = self.lib.load_network
        self.load_net.argtypes = [c_char_p, c_char_p, c_int]
        self.load_net.restype = c_void_p

        self.load_net_custom = self.lib.load_network_custom
        self.load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
        self.load_net_custom.restype = c_void_p

        self.do_nms_obj = self.lib.do_nms_obj
        self.do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        self.do_nms_sort = self.lib.do_nms_sort
        self.do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        self.free_image = self.lib.free_image
        self.free_image.argtypes = [IMAGE]

        self.letterbox_image = self.lib.letterbox_image
        self.letterbox_image.argtypes = [IMAGE, c_int, c_int]
        self.letterbox_image.restype = IMAGE

        self.load_meta = self.lib.get_metadata
        self.lib.get_metadata.argtypes = [c_char_p]
        self.lib.get_metadata.restype = METADATA

        self.load_image = self.lib.load_image_color
        self.load_image.argtypes = [c_char_p, c_int, c_int]
        self.load_image.restype = IMAGE

        self.rgbgr_image = self.lib.rgbgr_image
        self.rgbgr_image.argtypes = [IMAGE]

        self.predict_image = self.lib.network_predict_image
        self.predict_image.argtypes = [c_void_p, IMAGE]
        self.predict_image.restype = POINTER(c_float)

        self.predict_image_letterbox = self.lib.network_predict_image_letterbox
        self.predict_image_letterbox.argtypes = [c_void_p, IMAGE]
        self.predict_image_letterbox.restype = POINTER(c_float)

        self.network_predict_batch = self.lib.network_predict_batch
        self.network_predict_batch.argtypes = [c_void_p, IMAGE, c_int, c_int, c_int,
                                          c_float, c_float, POINTER(c_int), c_int, c_int]
        self.network_predict_batch.restype = POINTER(DETNUMPAIR)

        self.mode = 0

    def load(self, cfg,weights,meta):
        self.weights = weights
        self.net = self.load_net_custom(cfg.encode("ascii"), weights.encode("ascii"), 1, 1)  # batch size = 1
        self.meta = self.load_meta(meta.encode("ascii"))
        # In Python 3, the metafile default access craps out on Windows (but not Linux)
        # Read the names file and create a list to feed to detect
        try:
            with open(meta) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents, re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            self.names = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
        print("Initialized detector")


    def network_width(self):
        return self.lib.network_width(self.net)

    def network_height(self):
        return self.lib.network_height(self.net)

    def performBatchDetect(self, net, meta, image_list, pred_height, pred_width, c, thresh=0.25, hier_thresh=.5, nms=.45,
                           batch_size=3):
        self.mode = 1
        net_width, net_height = (self.network_width(net), self.network_height(net))
        img_list = []
        for custom_image_rgb in image_list:
            # custom_image = custom_image_bgr #cv2.cvtColor(custom_image_bgr, cv2.COLOR_BGR2RGB)
            custom_image = cv2.resize(
                custom_image_rgb, (net_width, net_height), interpolation=cv2.INTER_NEAREST)
            # custom_image = custom_image.transpose(2, 0, 1)
            # img_list.append(custom_image)
            img_list.append(custom_image.transpose(2, 0, 1))

        arr = np.concatenate(img_list, axis=0)
        arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
        data = arr.ctypes.data_as(POINTER(c_float))
        im = IMAGE(net_width, net_height, c, data)

        batch_dets = self.network_predict_batch(net, im, batch_size, pred_width,
                                           pred_height, thresh, hier_thresh, None, 0, 0)
        batch_boxes = []
        batch_scores = []
        batch_classes = []
        for b in range(batch_size):
            num = batch_dets[b].num
            dets = batch_dets[b].dets
            if nms:
                self.do_nms_obj(dets, num, meta.classes, nms)
            boxes = []
            scores = []
            classes = []
            for i in range(num):
                det = dets[i]
                score = -1
                label = None
                for c in range(det.classes):
                    p = det.prob[c]
                    if p > score:
                        score = p
                        label = c
                if score > thresh:
                    box = det.bbox
                    left, top, right, bottom = map(int, (box.x - box.w / 2, box.y - box.h / 2,
                                                         box.x + box.w / 2, box.y + box.h / 2))
                    boxes.append([left, top, right, bottom])
                    scores.append(score)
                    classes.append(label)

            batch_boxes.append(boxes)
            batch_scores.append(scores)
            batch_classes.append(classes)
        self.free_batch_detections(batch_dets, batch_size)
        self.mode = 0
        return batch_boxes, batch_scores, batch_classes

    def detect(self, image, thresh=.5, hier_thresh=.5, nms=.45, debug=False):
        """
        Performs the meat of the detection
        """
        # pylint: disable= C0321
        im = image
        if debug: print("Loaded image")
        ret = self.detect_image(im, thresh, hier_thresh, nms, debug)
        # free_image(im)
        if debug: print("freed image")
        return ret

    def detect_image(self, im, thresh=.5, hier_thresh=.5, nms=.45, debug=False):
        num = c_int(0)
        if debug: print("Assigned num")
        pnum = pointer(num)
        if debug: print("Assigned pnum")
        self.predict_image(self.net, im)
        letter_box = 0
        # predict_image_letterbox(net, im)
        # letter_box = 1
        if debug: print("did prediction")
        ##dets = get_network_boxes(net, custom_image_bgr.shape[1], custom_image_bgr.shape[0], thresh, hier_thresh, None, 0, pnum, letter_box) # OpenCV
        dets = self.get_network_boxes(self.net, im.w, im.h, thresh, hier_thresh, None, 0, pnum, letter_box)
        if debug: print("Got dets")
        num = pnum[0]
        if debug: print("got zeroth index of pnum")
        if nms:
            self.do_nms_sort(dets, num, self.meta.classes, nms)
        if debug: print("did sort")
        res = []
        if debug: print("about to range")
        for j in range(num):
            if debug: print("Ranging on " + str(j) + " of " + str(num))
            if debug: print("Classes: " + str(self.meta), self.meta.classes, self.meta.names)
            for i in range(self.meta.classes):
                if debug: print("Class-ranging on " + str(i) + " of " + str(self.meta.classes) + "= " + str(dets[j].prob[i]))
                if dets[j].prob[i] > 0:
                    b = dets[j].bbox
                    if self.names is None:
                        nameTag = self.meta.names[i]
                    else:
                        nameTag = self.names[i]
                    if debug:
                        print("Got bbox", b)
                        print(nameTag)
                        print(dets[j].prob[i])
                        print((b.x, b.y, b.w, b.h))
                    res.append((nameTag, dets[j].prob[i], (b.x, b.y, b.w, b.h)))
        if debug: print("did range")
        res = sorted(res, key=lambda x: -x[1])
        if debug: print("did sort")
        self.free_detections(dets, num)
        if debug: print("freed detections")
        return res