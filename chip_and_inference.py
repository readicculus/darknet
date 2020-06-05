import cv2
import json
import os

import requests
from darknet import performBatchDetect

dn_configs = "/home/yuval/Documents/XNOR/sealnet/models/darknet/cfg/EO/3_class/ensemble416x416/yolov3-tiny_3l.cfg"
dn_weights = "/fast/generated_data/PB-S_0/backup/saved/yolov3-tiny_3l_best.weights"

img_list_fn = "/data2/2019/all_rgb_images.txt"
dim = 832

img_list = open(img_list_fn).read().split('\n')
img_list = list(line for line in img_list if line)  #remove possible empty lines

host = "http://127.0.0.1:5000/api/image_chips?w={0}&h={0}&image_name={1}"
chip_dim = 832
for img_fn in img_list:
    img_base = os.path.basename(img_fn)
    get_uri = host.format(chip_dim, img_base)
    r = requests.get(url=get_uri)
    if r.status_code != 200:
        print("ERROR %s\n%s" % (img_base, r.text))
    json_res = json.loads(r.text)
    im = cv2.imread(img_fn)
    performBatchDetect()
    x=1
