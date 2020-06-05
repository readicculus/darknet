# Given experiment and list of images, crop images and run inference

# EXPERIMENT_NAME = 'yolov4_3c_608'
import argparse
import json
import os

from experiment_models import Experiment

EXPERIMENT_NAME = 'yolov4_3c_832'
ANIMAL_IMAGE_PATH = '/fast/generated_data/dataset_2/'
FP_IMAGE_PATH = '/fast/generated_data/fp_chips/832x832/'

parser = argparse.ArgumentParser(description='Train darknet.')
parser.add_argument('params', nargs='*')
args = parser.parse_args()
# data, cfg, weights = args.params
exp = Experiment('/fast/experiments/', EXPERIMENT_NAME)
session_name = 'base'
session = exp.sessions[session_name]
train_chips = session.df.get_train_images()
test_chips = session.df.get_test_images()

# split path to get full image location
def get_base_fn(chip_fn):
    fn_base, chip_coords = chip_fn.split('--')
    x1, y1, x2, y2 = chip_coords.replace('-', '_').split('_')
    return fn_base, chip_coords, x1, y1, x2, y2

jsons = []
for chip in train_chips:
    if chip == '':
        continue
    dir, chip_fn = os.path.split(chip)
    if 'fp' in dir:
        continue
    fn_noext = os.path.splitext(chip_fn)[0]
    json_fn = os.path.join(dir, fn_noext + '.json')
    if not os.path.isfile(json_fn):
        print('%s not found' % json_fn)
    else:
        jsons.append(json_fn)

# get all chips with animals
# key is the image id, value is array of chip id's containing animals
animal_img_to_chip_dict = {}
for jsonfile in jsons:
    with open(jsonfile) as f:
        arr = json.loads(f.read())
        for label in arr:
            im_id = label['global_label']['image_id']
            chip_id = label['relative_label']['chip_id']
            if not im_id in animal_img_to_chip_dict:
                animal_img_to_chip_dict[im_id] = set()
            animal_img_to_chip_dict[im_id].add(chip_id)
a=1