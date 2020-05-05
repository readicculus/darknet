import os

from experiment_models import Experiment


def count_classes(label_list, names):
    label_ct_by_class = [0] * len(names)
    bg_ct = 0
    fp_ct = 0
    truth_ct = 0
    with open(label_list) as file_in:
        for line in file_in:
            img_name, ext = os.path.splitext(line.strip())
            with open(img_name + '.txt') as label_file:
                line_ct = 0
                for l2 in label_file:
                    line_ct += 1
                    class_id = int(l2.split(' ')[0])
                    label_ct_by_class[class_id] += 1

                if line_ct > 0:
                    truth_ct += 1
                elif 'fp_chips' in img_name:
                    fp_ct += 1
                elif 'background_chips' in img_name:
                    bg_ct += 1

    res = {
        'image_type_counts': {
            'background': bg_ct,
            'false_positive': fp_ct,
            'truth': truth_ct
        },
        'class_counts': {

        }
    }
    for i in range(len(names)):
        res['class_counts'][names[i]] = label_ct_by_class[i]
    return res

def print_info(info, title):
    print(title)
    print("Class distribution")
    total = sum(list(info['class_counts'].values()))
    for k in info['class_counts']:
        print('%s : %d or %.2f%% of total' % (k, info['class_counts'][k], info['class_counts'][k]/total*100))
    print('total : %d Labels' % total)

    print('\nImage type distribution')
    total = sum(list(info['image_type_counts'].values()))
    for k in info['image_type_counts']:
        print('%s : %d or %.2f%% of total' % (k, info['image_type_counts'][k], info['image_type_counts'][k]/total*100))
    print('total : %d Images' % total)
    print('\n\n')

exp = Experiment('/fast/experiments/', 'yolov4_3c_608')
session= exp.sessions['2020-05-04_15-54-50']
df = session.get_datafile()
names = df.get_names()
train_info = count_classes(df.f_train,names)
test_info = count_classes(df.f_test,names)

print_info(train_info, "Train Breakdown")
print_info(test_info, "Test Breakdown")
# for s_name in exp.sessions:
#     session = exp.sessions[s_name]
#     session.get_session_breakdown()