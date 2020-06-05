import argparse

from experiment_info import count_classes, print_info
from experiment_models import Experiment


# EXPERIMENT_NAME = 'yolov4_3c_608'
EXPERIMENT_NAME = 'yolov4_3c_832'

parser = argparse.ArgumentParser(description='Train darknet.')
parser.add_argument('params', nargs='*')
args = parser.parse_args()
# data, cfg, weights = args.params
exp = Experiment('/fast/experiments/', EXPERIMENT_NAME)


def continue_existing_session(session_name, weights):
    exp.sessions[session_name].run(pretrained_weights=weights)

def start_session(session_name, weights):
    exp.sessions[session_name].run(pretrained_weights=weights, clear=True)


def new_session(data, cfg, weights, copy_pretrained_weights=True, name=None):
    return exp.new_session(data, cfg, weights, copy_pretrained_weights=copy_pretrained_weights, name=name)

def info(name):
    session = exp.sessions[name]
    df = session.get_datafile()
    names = df.get_names()
    train_info = count_classes(df.f_train, names)
    test_info = count_classes(df.f_test, names)

    print_info(train_info, "Train Breakdown")
    print_info(test_info, "Test Breakdown")

# session = new_session(data, cfg, weights,name='base')
# info('base')
# start_session('base', '/fast/experiments/yolov4_3c_832/base/weights/yolov4.conv.137')
continue_existing_session('base', '/fast/experiments/yolov4_3c_832/base/weights/yolov4-custom_last.weights')