import scipy.misc
import cv2
from csvloader import SealDataset
from darknet import performDetect, array_to_image
import os
import warnings

from utils import get_git_revisions_hash, Timer
import pickle
import argparse
import numpy as np
from darknet import lib
parser = argparse.ArgumentParser(description='Process images for new dataset')
parser.add_argument('-c', '--config', dest='config_path', required=True)
parser.add_argument('-data', '--data', dest='data_file', required=True)
parser.add_argument('-t', '--thresh', dest='thresh', default=.5)
parser.add_argument('-d', '--debug',  default=False, action='store_true')

dn_configs = ["/home/yuval/Documents/XNOR/sealnet/models/darknet/cfg/EO/3_class/ensemble416x416/yolov3-tiny_3l.cfg"]
dn_weights = ["/fast/generated_data/PB-S_0/backup/saved/yolov3-tiny_3l_best.weights"]

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

netMain = None
for dn_config, dn_weight in zip(dn_configs, dn_weights):

    netMain = performDetect("image.jpg", thresh=args.thresh, configPath=dn_config,
                  weightPath=dn_weight, metaPath=args.data_file, showImage=False, makeImageOnly=False,
                  initOnly=True)  ## initialize weight



    timer = Timer(len(test_dataset))
    time_remaining = 0
    for i, hs in enumerate(test_dataset):
        if i != 0 and i % 10 == 0:
            time_remaining = timer.remains(i)
        print("%.3f%% complete, %s" % (i / len(test_dataset) * 100, time_remaining), sep='', end='\r', flush=True)
        labels = hs["labels"]
        boxes = hs["boxes"]
        image = hs["image"]
        image = np.asarray(image)
        custom_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        custom_image = cv2.resize(custom_image,(lib.network_width(netMain), lib.network_height(netMain)), interpolation = cv2.INTER_LINEAR)
        im, arr = array_to_image(custom_image)
        dets = performDetect(im, thresh=args.thresh, configPath=dn_config,
                      weightPath=dn_weight, metaPath=args.data_file, showImage=False, makeImageOnly=False,
                      initOnly=False)  ## initialize weight
        print(dets)