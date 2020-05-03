import argparse
import glob
import logging
import os
import signal
import subprocess
import atexit
import sys

from datetime import datetime
from io import StringIO
from shutil import copyfile


def try_mkdir(path):
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)


class DataFile(object):
    def __init__(self, n_classes=None, f_train=None, f_valid=None, f_test=None, f_names=None, dir_backup=None):
        self.n_classes = n_classes
        self.f_train = f_train
        self.f_test = f_test
        self.f_valid = f_valid
        self.f_names = f_names
        self.dir_backup = dir_backup

    def load(self, file):
        with open(file) as f:
            vars = {}
            for line in f:
                name, var = line.split('=')
                vars[name.strip()] = var.strip()

            self.n_classes = int(vars['classes'])
            self.f_train = vars['train']
            self.f_test = vars['test']
            self.f_valid = vars['valid']
            self.f_names = vars['names']
            self.dir_backup = vars['backup']

    def migrate_file(self, new_dir, file, name):
        new_file = os.path.join(new_dir, name)
        copyfile(file, new_file)
        return new_file

    def migrate(self, new_dir):
        test_valid_same = (self.f_test == self.f_valid)
        self.f_train = self.migrate_file(new_dir, self.f_train, 'train.txt')
        self.f_test = self.migrate_file(new_dir, self.f_test, 'test.txt')
        if test_valid_same:
            self.f_valid = self.f_test
        else:
            self.f_valid = self.migrate_file(new_dir, self.f_valid, 'valid.txt')
        self.f_names = self.migrate_file(new_dir, self.f_names, 'names.txt')
        self.dir_backup = os.path.join(new_dir, os.path.split(self.dir_backup)[1])
        try_mkdir(self.dir_backup)

    def save(self, file):
        try:
            with open(file, "w") as f:
                f.write('classes = %d\n' % self.n_classes)
                f.write('train = %s\n' % self.f_train)
                f.write('test = %s\n' % self.f_test)
                f.write('valid = %s\n' % self.f_valid)
                f.write('names = %s\n' % self.f_names)
                f.write('backup = %s' % self.dir_backup)
        except:
            return False
        return True


class TrainSession(object):
    def __init__(self, name=None, exp_dir=None, data=None, cfg=None, pretrained_weights=None, copy_pretrained_weights=None):
        if not all(v is None for v in [name,exp_dir,data,cfg,pretrained_weights,copy_pretrained_weights]):
            self.name = name
            self.dir = os.path.join(exp_dir, self.name)
            try_mkdir(self.dir)
            self.output_file = os.path.join(self.dir, 'output.txt')

            self.data_in = data
            self.data_out = os.path.join(self.dir, 'yolo.data')

            self.df = DataFile()
            self.df.load(self.data_in)  # load the given .data file
            self.df.migrate(self.dir)  # migrate all the training files to here so we can know what was used
            self.df.save(self.data_out)  # save the new training file to the train session directory

            self.cfg_in = os.path.join(os.getcwd(), cfg)
            self.cfg_out = os.path.join(self.dir, os.path.split(self.cfg_in)[1])
            self.pretrained_weights_in = pretrained_weights
            self.pretrained_weights_out = os.path.join(self.df.dir_backup, os.path.split(self.pretrained_weights_in)[1])
            if copy_pretrained_weights:
                copyfile(self.pretrained_weights_in, self.pretrained_weights_out)
            else:
                self.pretrained_weights_out = self.pretrained_weights_in

            copyfile(self.cfg_in, self.cfg_out)

    def run(self):
        params = [self.data_out, self.cfg_out, self.pretrained_weights_out]
        arg_arr = ['./darknet'] + ["detector", "train"] + params + ["-gpus"] + ['1'] + ['-map']
        cmd = " ".join(arg_arr)
        f = open(self.output_file, "a")
        f.write(cmd)
        print(cmd, end='')

        from subprocess import Popen, PIPE, STDOUT

        with Popen(arg_arr, stdout=PIPE, stderr=STDOUT, bufsize=1,
                   universal_newlines=True) as p, StringIO() as buf:
            for line in p.stdout:
                print(line, end='')
                f.write(line)
                buf.write(line)
            output = buf.getvalue()
        rc = p.returncode

        f.close()

    def load(self, dir):
        self.data_in = self.data_out = os.path.join(dir, 'yolo.data')
        self.output_file = os.path.join(dir, 'output.txt')
        self.cfg_in = self.cfg_out = glob.glob(os.path.join(dir,'*.cfg'))[0]
        self.df = DataFile()
        self.df.load(os.path.join(self.data_in))


class Experiment(object):
    def __init__(self, dir, name):
        self.dir = os.path.join(dir, name)
        self.name = name
        self.dataset = None
        self.pretrained_weights = None
        self.config = None
        self.sessions = []

        if os.path.exists(self.dir):
            self.load()
        else:
            # make experiment directory
            try_mkdir(self.dir)

    def load(self):
        subdirs = os.listdir(self.dir)
        for sd in subdirs:
            ts = TrainSession()
            ts.load(os.path.join(self.dir, sd))
            self.sessions.append(ts)


    def new_session(self, data, cfg, pretrained_weights, copy_pretrained_weights=False):
        now = datetime.now()
        sess_name = now.strftime("%Y-%m-%d_%H-%M-%S")
        session = TrainSession(sess_name, self.dir, data, cfg, pretrained_weights, copy_pretrained_weights)
        self.sessions.append(session)
        return session


parser = argparse.ArgumentParser(description='Train darknet.')
parser.add_argument('params', nargs='*')
args = parser.parse_args()
data, cfg, weights = args.params

exp = Experiment('/fast/experiments/', 'yolov4_3c_608')
session = exp.new_session(data, cfg, weights)
session.run()
