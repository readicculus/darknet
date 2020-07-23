
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

    def __repr__(self):
        res = "%d %.4f (x1:%d,y1:%d) (x2:%d,y2:%d)" % (self.classid, self.confidence,
                                                       self.box[0], self.box[1],self.box[2],self.box[3])
        return res

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