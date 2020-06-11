from concurrent.futures import ThreadPoolExecutor
from pydn.inference.darknet import *

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

class Detector(object):
    def __init__(self, config_fn, meta_fn, weights_fn):
        self.config_fn = config_fn
        self.meta_fn = meta_fn
        self.weights_fn = weights_fn


class GPUInferenceBatch():
    def __init__(self, gpu, im_batch, meta_batch, dim, nbatch):
        self.gpu = gpu
        self.im_batch = im_batch
        self.meta_batch = meta_batch
        self.dim = dim
        self.batch_boxes, self.batch_scores, self.batch_classes = None,None,None
        self.nbatch = nbatch

    def set_results(self, batch_boxes, batch_scores, batch_classes):
        self.batch_boxes = batch_boxes
        self.batch_scores = batch_scores
        self.batch_classes = batch_classes

class BatchDetector():
    def __init__(self, net, meta, batch_size):
        self.net = net
        self.meta = meta
        self.batch_size = batch_size

class MultiGPUBatchDetectors():
    def __init__(self, config, meta, weights, dim, gpus=[0,1], batch_sizes=[1,1],
                 nms_thresh = .45, hier_thresh=.5, thresh=.25):
        assert (len(gpus) == len(batch_sizes))
        self.config = config
        self.meta = meta
        self.weights = weights
        self.num_gpu = len(gpus)
        self.total_batch_size = sum(batch_sizes)
        self.nms_thresh = nms_thresh
        self.hier_thresh = hier_thresh
        self.thresh = thresh
        self.dim = dim

        self.detectors = {}
        self.image_queue = []
        for gpu_i, batch_size_i in zip(gpus, batch_sizes):
            set_gpu(gpu_i)  # running on GPU i
            net_i, meta_i = load_net_meta_batch(configPath=self.config, weightPath=self.weights, metaPath=self.meta,
                                              batchSize=batch_size_i)
            detector_i = BatchDetector(net_i, meta_i, batch_size_i)
            self.detectors[gpu_i] = detector_i

    def batch_detect(self, gbatch):
        assert (gbatch.gpu in self.detectors)
        batch_detector = self.detectors[gbatch.gpu]
        batch_boxes, batch_scores, batch_classes = performBatchDetect(batch_detector.net, batch_detector.meta, image_list=gbatch.im_batch, pred_height=gbatch.dim,
                           pred_width=gbatch.dim,
                           c=3,
                           batch_size=batch_detector.batch_size,
                           thresh=self.thresh,
                           hier_thresh=self.hier_thresh,
                           nms=self.nms_thresh)
        gbatch.set_results(batch_boxes, batch_scores, batch_classes)
        return gbatch


    def detect(self, image_list):
        for im in image_list:
            self.image_queue.append(im)

        batch_results = []
        while (len(self.image_queue) >= self.total_batch_size):
            gbatches = []
            for gpu_i in self.detectors:
                batch_meta = []
                batch_chips = []
                detector = self.detectors[gpu_i]
                for _ in range(detector.batch_size):
                    dn_chip = self.image_queue.pop()
                    batch_meta.append(dn_chip.chip_meta)
                    batch_chips.append(dn_chip.chip_im)
                gbatch = GPUInferenceBatch(gpu_i, batch_chips, batch_meta, self.dim, detector.batch_size)
                gbatches.append(gbatch)

            with ThreadPoolExecutor() as executor:
                batch_results += [val for val in executor.map(self.batch_detect, gbatches)]
        return batch_results