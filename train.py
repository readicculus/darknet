import argparse


from experiment_models import Experiment




parser = argparse.ArgumentParser(description='Train darknet.')
parser.add_argument('params', nargs='*')
args = parser.parse_args()
data, cfg, weights = args.params

def run_existing_session():
    exp = Experiment('/fast/experiments/', 'yolov4_3c_608')
    exp.sessions['2020-05-04_15-54-50'].run(pretrained_weights='/fast/experiments/yolov4_3c_608/2020-05-04_15-54-50/weights/yolov4-custom_last.weights')

def new_session(clear=True, copy_pretrained_weights=True):
    exp = Experiment('/fast/experiments/', 'yolov4_3c_608')
    session = exp.new_session(data, cfg, weights, copy_pretrained_weights=copy_pretrained_weights)
    session.run(clear=clear)


run_existing_session()