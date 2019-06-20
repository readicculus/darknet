import argparse
import logging
import os
import signal
import subprocess
import atexit
import sys
from io import StringIO

parser = argparse.ArgumentParser(description='Train darknet.')
parser.add_argument('params', nargs='*')
args = parser.parse_args()
params = args.params

arg_arr = ['./darknet'] + ["detector", "train"] + params + [" -gpus 0"]
cmd = " ".join(arg_arr)
# result = subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)


from subprocess import Popen, PIPE, STDOUT
with Popen(arg_arr, stdout=PIPE, stderr=STDOUT, bufsize=1,
           universal_newlines=True) as p, StringIO() as buf:
    for line in p.stdout:
        print(line, end='')
        buf.write(line)
    output = buf.getvalue()
rc = p.returncode


# for line in iter(process.stdout.readline, ''):  # replace '' with b'' for Python 3
#         sys.stdout.write(line)
# exitcode = process.wait() # 0 means success
