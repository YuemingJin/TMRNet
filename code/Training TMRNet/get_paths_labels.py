import os
import numpy as np
import pickle

root_dir2 = '../../data/cholec80/'
img_dir2 = os.path.join(root_dir2, 'cutMargin')
phase_dir2 = os.path.join(root_dir2, 'phase_annotations')

#train_video_num = 8
#val_video_num = 4
#test_video_num = 2

print(root_dir2)
print(img_dir2)
print(phase_dir2)


#cholec80==================
def get_dirs2(root_dir):
    file_paths = []
    file_names = []
    for lists in os.listdir(root_dir):
        path = os.path.join(root_dir, lists)
        if os.path.isdir(path):
            file_paths.append(path)
            file_names.append(os.path.basename(path))
    file_names.sort(key=lambda x:int(x))
    file_paths.sort(key=lambda x:int(os.path.basename(x)))
    return file_names, file_paths

def get_files2(root_dir):
    file_paths = []
    file_names = []
    for lists in os.listdir(root_dir):
        path = os.path.join(root_dir, lists)
        if not os.path.isdir(path):
            file_paths.append(path)
            file_names.append(os.path.basename(path))
    file_names.sort()
    file_paths.sort()
    return file_names, file_paths
#cholec80==================


#cholec80==================
img_dir_names2, img_dir_paths2 = get_dirs2(img_dir2)
phase_file_names2, phase_file_paths2 = get_files2(phase_dir2)

phase_dict = {}
phase_dict_key = ['Preparation', 'CalotTriangleDissection', 'ClippingCutting', 'GallbladderDissection',
                  'GallbladderPackaging', 'CleaningCoagulation', 'GallbladderRetraction']
for i in range(len(phase_dict_key)):
    phase_dict[phase_dict_key[i]] = i
print(phase_dict)
#cholec80==================


#cholec80==================
all_info_all2 = []

for j in range(len(phase_file_names2)):
    downsample_rate = 25
    phase_file = open(phase_file_paths2[j])

    video_num_file = int(os.path.splitext(os.path.basename(phase_file_paths2[j]))[0][5:7])
    video_num_dir = int(os.path.basename(img_dir_paths2[j]))

    print("video_num_file:", video_num_file,"video_num_dir:", video_num_dir, "rate:", downsample_rate)

    info_all = []
    first_line = True
    for phase_line in phase_file:
        phase_split = phase_line.split()
        if first_line:
            first_line = False
            continue
        if int(phase_split[0]) % downsample_rate == 0:
            info_each = []
            img_file_each_path = os.path.join(img_dir_paths2[j], phase_split[0] + '.jpg')
            info_each.append(img_file_each_path)
            info_each.append(phase_dict[phase_split[1]])
            info_all.append(info_each)              

    # print(len(info_all))
    all_info_all2.append(info_all)
#cholec80==================

with open('./cholec80.pkl', 'wb') as f:
    pickle.dump(all_info_all2, f)

with open('./cholec80.pkl', 'rb') as f:
    all_info_80 = pickle.load(f)

#cholec80==================
train_file_paths_80 = []
test_file_paths_80 = []
val_file_paths_80 = []
val_labels_80 = []
train_labels_80 = []
test_labels_80 = []

train_num_each_80 = []
val_num_each_80 = []
test_num_each_80 = []

for i in range(32):
    train_num_each_80.append(len(all_info_80[i]))
    for j in range(len(all_info_80[i])):
        train_file_paths_80.append(all_info_80[i][j][0])
        train_labels_80.append(all_info_80[i][j][1:])

print(len(train_file_paths_80))
print(len(train_labels_80))

for i in range(32,40):
    val_num_each_80.append(len(all_info_80[i]))
    for j in range(len(all_info_80[i])):
        val_file_paths_80.append(all_info_80[i][j][0])
        val_labels_80.append(all_info_80[i][j][1:])

print(len(val_file_paths_80))
print(len(val_labels_80))


#cholec80==================


train_val_test_paths_labels = []
train_val_test_paths_labels.append(train_file_paths_80)
train_val_test_paths_labels.append(val_file_paths_80)

train_val_test_paths_labels.append(train_labels_80)
train_val_test_paths_labels.append(val_labels_80)

train_val_test_paths_labels.append(train_num_each_80)
train_val_test_paths_labels.append(val_num_each_80)

with open('train_val_paths_labels.pkl', 'wb') as f:
    pickle.dump(train_val_test_paths_labels, f)


print('Done')
print()
