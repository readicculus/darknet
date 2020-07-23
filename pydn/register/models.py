import glob
import os

import cv2
import h5py

import numpy as np

left_h5_f = '/home/yuval/Documents/XNOR/sealnet/models/darknet/pydn/register/output_transform_Left.h5'
cent_h5_f = '/home/yuval/Documents/XNOR/sealnet/models/darknet/pydn/register/output_transform_4Center.h5'

class RegistrationHelper():
    def __init__(self, h5_file, inverse=False):
        self.h5_file = h5_file
        self.H = self.get_homography(inverse=inverse)
        
    def get_homography(self, warp_mode = cv2.MOTION_EUCLIDEAN, inverse=False):
        with h5py.File(self.h5_file, "r") as f:
            # List all groups
            data = f['/']['TransformGroup']['0']['TransformParameters']
            affine=np.array(data.value)
            a00,a10,a01,a11,a02,a12=affine
            H = None
            if warp_mode == cv2.MOTION_HOMOGRAPHY :
                H = np.array([[a00,a01,a02], [a10,a11,a12], [0,0,1]]).astype(np.double)
                if inverse:
                    H = np.linalg.inv(H)
            else :
                H = np.array([[a00,a01,a02], [a10,a11,a12]]).astype(np.double)
            return H

    
    def register_image(self, rgb_name):
        pass

    def project_point(self, point, S=None):
        x,y = point
        z=1
        pt=np.array((x,y,1)).reshape((3, 1))
        p_pt = self.H.dot(pt)
        sum = np.sum(p_pt, 1)
        px = p_pt[0] / sum[2]
        py = p_pt[1] / sum[2]
        p_pt = np.array((px,py))
        # transformed = cv2.perspectiveTransform(points, self.homography)

        return p_pt

    def project_box(self, box, x_scalar=None, y_scalar=None):
        x1,y1,x2,y2 = box
        S =  None

        if x_scalar is not None and y_scalar is not None:
            # x1 = x1*x_scalar
            # x2 = x2*x_scalar
            # y1 = y1 * y_scalar
            # y2 = y2 * y_scalar
            S = np.zeros((3,3))
            S[0][0] = x_scalar
            S[1][1] = x_scalar
            S[2][2] = 1

        output = cv2.perspectiveTransform(np.array([[[x1, y1], [x2,y2], [x1,y2], [x2,y1]]], dtype="float32"), self.H)
        x1 = min(output[0][:,0])
        y1 = min(output[0][:,1])
        x2 = max(output[0][:, 0])
        y2 = max(output[0][:, 1])
        # print(output[0])

        # (x1,y1) = self.project_point((x1,y1), S)
        # (x2,y2) = self.project_point((x2,y2), S)
        # print(((x1,y1), (x2,y2)))
        # print()


        return ((x1,y1), (x2,y2))

def get_file_pairs(image_dir):
    fn_filter = os.path.join(image_dir, "*_rgb.jpg")
    rgb_files = glob.glob(fn_filter)
    ir_files = []
    for file in rgb_files:
        ir_fn = '_'.join(file.split("_")[:-1]) + '_ir.tif'
        if os.path.isfile(ir_fn):
            ir_files.append(ir_fn)
        else:
            ir_files.append(None)

    im_pairs = {a:b for a,b in zip(rgb_files, ir_files)}
    return im_pairs

