import glob
import os
import re
import sys
import logging
from datetime import datetime
from shutil import copyfile
from log import DarknetLogParser


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

    def get_names(self):
        names = []
        with open(self.f_names) as f:
            for line in f:
                names.append(line.strip())
        return names

    def get_train_images(self):
        return open(self.f_train).read().split('\n')

    def get_test_images(self):
        return open(self.f_test).read().split('\n')

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
        self.file_logger = logging.getLogger('TrainLog')
        self.file_logger.setLevel(logging.INFO)
        self.console_logger = logging.getLogger('ConsoleLog')
        self.console_logger.setLevel(logging.INFO)
        self.log_parser = DarknetLogParser()

        self.name = name
        self.pretrained_weights_out = self.pretrained_weights_in = None
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

    def get_datafile(self):
        return self.df

    def run(self, calc_map=True, gpus=[1], pretrained_weights = None, clear=False):
        if self.pretrained_weights_out is None:
            self.pretrained_weights_out = self.pretrained_weights_in = pretrained_weights
        self.start_logging(self.output_file)
        params = [self.data_out, self.cfg_out, self.pretrained_weights_out]
        arg_arr = ['./darknet'] + ["detector", "train"] + params

        if not gpus is None and len(gpus) > 0:
            arg_arr = arg_arr + ["-gpus"]
            gpus = [str(gpu) for gpu in gpus]
            arg_arr = arg_arr + [','.join(gpus)]

        if calc_map: arg_arr = arg_arr + ['-map']
        if clear: arg_arr = arg_arr + ['-clear']
        arg_arr = arg_arr + ['-mAP_epochs'] + ['2']
        cmd = " ".join(arg_arr)
        from subprocess import Popen, PIPE, STDOUT
        self.log("Beginning training session:")
        self.log(cmd)
        sys.stderr = open('out.log', 'w')
        with Popen(arg_arr, stdout=PIPE, stderr=STDOUT,
                   universal_newlines=True, bufsize=0) as p:
            for line in iter(p.stdout.readline, ''):
                self.log(line.strip())
            # for line in p.stdout:
            #     self.log(line.strip())

        rc = p.returncode
        self.logger.disabled = True

    def load(self, dir):
        self.name = os.path.basename(os.path.normpath(dir))
        self.data_in = self.data_out = os.path.join(dir, 'yolo.data')
        self.output_file = os.path.join(dir, 'output.txt')
        self.cfg_in = self.cfg_out = glob.glob(os.path.join(dir,'*.cfg'))[0]
        self.df = DataFile()
        self.df.load(os.path.join(self.data_in))

    def start_logging(self, file):
        self.fh = logging.FileHandler(file)
        self.fh.setLevel(logging.INFO)
        self.file_logger.addHandler(self.fh)

        self.ch = logging.StreamHandler()
        self.ch.setLevel(logging.INFO)
        self.console_logger.addHandler(self.ch)

    def init_log(self):
        pass

    def log(self, s):
        self.console_logger.info(s)
        p = self.log_parser.parse(s)
        if not p is None:
            self.file_logger.info(p)

class Experiment(object):
    def __init__(self, dir, name):
        self.dir = os.path.join(dir, name)
        self.name = name
        self.dataset = None
        self.pretrained_weights = None
        self.config = None
        self.sessions = {}

        if os.path.exists(self.dir):
            self.load()
        else:
            # make experiment directory
            try_mkdir(self.dir)

    def load(self):
        subdirs = os.listdir(self.dir)
        for session_name in subdirs:
            ts = TrainSession()
            ts.load(os.path.join(self.dir, session_name))
            self.sessions[session_name] = ts


    def new_session(self, data, cfg, pretrained_weights, copy_pretrained_weights=False, name=None):
        now = datetime.now()
        if name is None:
            name = now.strftime("%Y-%m-%d_%H-%M-%S")
        session = TrainSession(name, self.dir, data, cfg, pretrained_weights, copy_pretrained_weights)
        self.sessions[name] = session
        return session