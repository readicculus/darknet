import pandas as pd
from asyncio import sleep
from multiprocessing.managers import BaseManager

import scipy.misc
import cv2
from csvloader import SealDataset
from darknet import array_to_image, Detector
import os
import warnings

from transforms.crops import full_image_tile_crops
from util import get_tile_images, Detection
from utils import get_git_revisions_hash, Timer
import pickle
import argparse
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import multiprocessing as mp
from multiprocessing import Pool, Queue

parser = argparse.ArgumentParser(description='Process images for new dataset')
parser.add_argument('-c', '--config', dest='config_path', required=True)
parser.add_argument('-data', '--data', dest='data_file', required=True)
parser.add_argument('-t', '--thresh', dest='thresh', default=.5, type=float)
parser.add_argument('-d', '--debug',  default=False, action='store_true')

dn_configs = ["/home/yuval/Documents/XNOR/sealnet/models/darknet/cfg/EO/3_class/ensemble416x416/yolov3-tiny_3l.cfg",
              "/home/yuval/Documents/XNOR/sealnet/models/darknet/cfg/EO/3_class/ensemble416x416/yolov3-tiny_2l.cfg",
              "/home/yuval/Documents/XNOR/sealnet/models/darknet/cfg/EO/3_class/ensemble416x416/yolov3-tiny_3l_foc_loss.cfg",
              "/home/yuval/Documents/XNOR/sealnet/models/darknet/cfg/EO/3_class/ensemble416x416/yolov3.cfg"]
dn_weights = ["/fast/generated_data/PB-S_0/backup/saved/yolov3-tiny_3l_best.weights",
              "/fast/generated_data/PB-S_0/backup/saved/yolov3-tiny_2l_best.weights",
              "/fast/generated_data/PB-S_0/backup/saved/yolov3-tiny_3l_foc_loss_best.weights",
              "/fast/generated_data/PB-S_0/backup/saved/yolov3_best.weights"]

args = parser.parse_args()
DEBUG = args.debug

# Load the configuration
config = None
try:
    pickle_file_path = args.config_path
    pickle_file = open(pickle_file_path,'rb')
    config = pickle.load(pickle_file)
except:
    raise Exception("Could not load file " + pickle_file_path)

current_hash = get_git_revisions_hash()
if current_hash != config.hash:
    warnings.warn("Current git hash is not equal to config git hash")


dataset_base = os.path.join(config.generated_data_base, config.dataset_path)
train_list = os.path.join(dataset_base, config.system.train_list)
test_list = os.path.join(dataset_base, config.system.test_list)
# Check required outline files exist
if not os.path.exists(dataset_base):
    raise Exception("specified dataset {} does not exist.".format(dataset_base))
if not os.path.isfile(train_list):
    raise Exception("specified train list {} does not exist.".format(train_list))
if not os.path.isfile(test_list):
    raise Exception("specified test list {} does not exist.".format(test_list))

# load train and test dataset
train_dataset = SealDataset(csv_file=train_list, root_dir='/data/raw_data/TrainingAnimals_ColorImages/')
test_dataset = SealDataset(csv_file=test_list, root_dir='/data/raw_data/TrainingAnimals_ColorImages/')
print("Generating %s set--------------------" % type)
plt.ion()



plt_im = None
fig = None


# detector = Detector()
detections = {}
df = None
detectors = []
i=0
for dn_config, dn_weight in zip(dn_configs, dn_weights):
    detector = Detector()
    detector.set_gpu(i%2)
    detector.load(dn_config, dn_weight, args.data_file)
    detectors.append(detector)
    i+=1

# for dn_config, dn_weight in zip(dn_configs, dn_weights):
#     detector.load(dn_config, dn_weight, args.data_file)

timer = Timer(len(test_dataset))
time_remaining = 0
# if not dn_config in detections:
#     detections[dn_config] = {}
for i, hs in enumerate(test_dataset):
    if i != 0 and i % 10 == 0:
        time_remaining = timer.remains(i)
    print("%.3f%% complete, %s" % (i / len(test_dataset) * 100, time_remaining), sep='', end='\r', flush=True)
    labels = hs["labels"]
    boxes = hs["boxes"]
    image = hs["image"]
    filename = os.path.basename(os.path.basename(image.filename))
    image = np.asarray(image)
    tiles = full_image_tile_crops(image, detector.network_width(),detector.network_height())
    image_dets = []
    for detector in detectors:
        weights_file_base = os.path.basename(os.path.basename(detector.weights))
        for tile, location in tiles:
            tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
            tile = cv2.resize(tile, (detector.network_width(), detector.network_height()),
                              interpolation=cv2.INTER_LINEAR)

            im, arr = array_to_image(tile)
            ds = detector.performDetect(im, thresh=args.thresh)

            if len(ds) > 0:

                for det in ds:
                    new = Detection(det[2][0], det[2][1], det[2][2], det[2][3], det[0], det[1])
                    new.shift(location[2], location[0])
                    image_dets.append(new)
        g= pd.DataFrame.from_records([s.to_dict(filename,weights_file_base) for s in image_dets])
        if df is None:
            df = g
            g.to_csv('test.csv', sep='\t', header=True, mode='w',index=False)
        else:
            df = df.append(g, ignore_index=True)
            g.to_csv('test.csv', sep='\t', header=False, mode='a',index=False)


    # detections[dn_config][filename] = image_dets
    print("%d detections in %s" % (len(image_dets), filename))
    # print(dets)
    # for d in dets:
    #     x1,x2,y1,y2 = d.x1x2y1y2()
    #     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    #
    # if plt_im is None:
    #     fig = plt.figure()
    #
    #     plt_im = plt.imshow(image, cmap='gist_gray_r')
    # else:
    #     plt_im.set_data(image)
    #     plt_im = plt.imshow(image, cmap='gist_gray_r')
    # plt.draw()
    # plt.show()
    # plt.pause(1)
    # plt.cla()
