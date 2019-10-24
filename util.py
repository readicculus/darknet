import numpy as np
def get_tile_images(image, width=8, height=8):
    _nrows, _ncols, depth = image.shape
    _size = image.size
    _strides = image.strides

    nrows, _m = divmod(_nrows, height)
    ncols, _n = divmod(_ncols, width)
    if _m != 0 or _n != 0:
        return None

    return np.lib.stride_tricks.as_strided(
        np.ravel(image),
        shape=(nrows, ncols, height, width, depth),
        strides=(height * _strides[0], width * _strides[1], *_strides),
        writeable=False
    )

class Detection(object):
    def __init__(self, x,y,w,h,label, confidence):
        self.x = int(x)
        self.y = int(y)
        self.w = int(w)
        self.h = int(h)
        self.label = label
        self.confidence = confidence

    def shift(self, dx, dy):
        self.x += int(dx)
        self.y += int(dy)

    def x1x2y1y2(self):
        hx = self.w / 2
        hy = self.h / 2
        return (int(self.x-hx), int(self.x+hx), int(self.y-hy), int(self.y+hy))

    def __str__(self):
        r = "%s %.4f [%d, %d, %d, %d]" % (self.label, self.confidence, self.x, self.y, self.w, self.h)
        return r

    def __repr__(self):
        return self.__str__()