import pickle
import shutil
import os
import argparse

with open('./cholec80.pkl', 'rb') as f:
    test_paths_labels = pickle.load(f)

parser = argparse.ArgumentParser(description='lstm testing')
parser.add_argument('-s', '--seq', default=10, type=int, help='sequence length, default 10')
parser.add_argument('-n', '--name', type=str, help='name of pred')

args = parser.parse_args()
sequence_length = args.seq
pred_name = args.name

with open(pred_name, 'rb') as f:
    ori_preds = pickle.load(f)

num_video = 40
num_labels = 0
for i in range(40,80):
    num_labels += len(test_paths_labels[i])

num_preds = len(ori_preds)

print('num of labels  : {:6d}'.format(num_labels))
print("num ori preds  : {:6d}".format(num_preds))
print("labels example : ", test_paths_labels[0][0][1])
print("preds example  : ", ori_preds[0])

if num_labels == (num_preds + (sequence_length - 1) * num_video):

    phase_dict_key = ['Preparation', 'CalotTriangleDissection', 'ClippingCutting', 'GallbladderDissection',
                  'GallbladderPackaging', 'CleaningCoagulation', 'GallbladderRetraction']
    preds_all = []
    label_all = []
    count = 0
    for i in range(40,80):
        filename = '../result/phase/video' + str(1 + i) + '-phase.txt'
        gt_filename = '../result/gt-phase/video' + str(1 + i) + '-phase.txt'
        f = open(filename, 'w')
        #f.write('Frame Phase')
        #f.write('\n')

        f2 = open(gt_filename, 'w')
        #f2.write('Frame Phase')
        #f2.write('\n')

        preds_each = []
        for j in range(count, count + len(test_paths_labels[i]) - (sequence_length - 1)):
            if j == count:
                # TODO 单个视频的初始几个帧的predict设置
                for k in range(sequence_length - 1):
                    preds_each.append(0)
                    preds_all.append(0)
            preds_each.append(ori_preds[j])
            preds_all.append(ori_preds[j])
        for k in range(len(preds_each)):
            f.write(str(25 * k))
            f.write('\t')
            f.write(str(int(preds_each[k])))
            f.write('\n')
            
            f2.write(str(25 * k))
            f2.write('\t')
            f2.write(str(int(test_paths_labels[i][k][1])))
            label_all.append(test_paths_labels[i][k][1])
            f2.write('\n')

        f.close()
        f2.close()
        count += len(test_paths_labels[i]) - (sequence_length - 1)
    test_corrects = 0

    print('num of labels       : {:6d}'.format(len(label_all)))
    print('rsult of all preds  : {:6d}'.format(len(preds_all)))

    for i in range(num_labels):
# TODO
        if int(label_all[i]) == int(preds_all[i]):
            test_corrects += 1

    print('right number preds  : {:6d}'.format(test_corrects))
    print('test accuracy       : {:.4f}'.format(test_corrects / num_labels))
else:
    print('number error, please check')
