import argparse
import os
import signal
import subprocess
import atexit
import sys

parser = argparse.ArgumentParser(description='Train darknet.')
parser.add_argument('params', nargs='*')
args = parser.parse_args()
params = args.params

cmd = " ".join(['./darknet'] + ["detector", "train"] + params)
# cmd = "./darknet detector train " + " ".join(params)
result = subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
# proc = subprocess.run(cmd)



