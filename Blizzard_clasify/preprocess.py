#!usr/bin/env python  
#-*- coding:utf-8 _*- 
"""
@version: python3.6
"""
from glob import glob
import os
import codecs
import random
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import shutil

# base_path = 'data/'
# data_path = base_path + 'garbage_classify/train_data'
data_path = '/data/multi_task/Ubuntu_Code/garbage_classify/huawei-garbage-master/Data'
label_files = glob(os.path.join(data_path, '*.txt'))
img_paths = []
labels = []
result = []
label_dict = {}
data_dict = {}
#
for index, file_path in enumerate(label_files):
    with codecs.open(file_path, 'r', 'utf-8') as f:
        line = f.readline()
    line_split = line.strip().split(', ')
    if len(line_split) != 2:
        print('%s contain error lable' % os.path.basename(file_path))
        continue
    img_name = line_split[0]
    label = int(line_split[1])
    img_paths.append(os.path.join(data_path, img_name))
    labels.append(label)
    result.append(os.path.join(data_path, img_name) + ',' + str(label))
    label_dict[label] = label_dict.get(label, 0) + 1
    if label not in data_dict:
        data_dict[label] = []
    data_dict[label].append(os.path.join(data_path, img_name) + ',' + str(label))


folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=2019)
for fold_, (trn_idx, val_idx) in enumerate(folds.split(result, labels)):
    train_data = list(np.array(result)[trn_idx])
    val_data = list(np.array(result)[val_idx])

print(len(train_data), len(val_data))

for train_img_label in train_data:
    # print(train_img_label)
    img_path, label = train_img_label.split(',')
    shutil.copy(img_path, '/data/multi_task/Ubuntu_Code/garbage_classify/huawei-garbage-master/DataSet/TrainData')
    label_path,_ = img_path.split('.')
    # print(label_path)
    label_path = label_path+'.txt'
    shutil.copy(label_path,'/data/multi_task/Ubuntu_Code/garbage_classify/huawei-garbage-master/DataSet/TrainData')
for test_img_label in val_data:
    # print(train_img_label)
    img_path, label = test_img_label.split(',')
    print(img_path)
    shutil.copy(img_path, '/data/multi_task/Ubuntu_Code/garbage_classify/huawei-garbage-master/DataSet/TestData')
    label_path, _ = img_path.split('.')
    # print(label_path)
    label_path = label_path + '.txt'
    shutil.copy(label_path, '/data/multi_task/Ubuntu_Code/garbage_classify/huawei-garbage-master/DataSet/TestData')
    print(label_path)
    
    
