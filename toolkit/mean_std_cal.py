#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This scripts calculates:
    1. mean&std of dataset
    2. eigenvector&eigenvalue of dataset

"""

import numpy as np
import os
from PIL import Image
import glob

img_h, img_w = 448, 448  # modify to specific size
imgs_path = '/data/multi_task/Ubuntu_Code/garbage_classify/Data'  # modify to specific dir
# imgs_path = '/home/yxh/DataSet/kaggle2015/train/'
# ---------- calculate mean and std of dataset ----------
means, stdevs = [], []
img_list = []
imgs_path_list = os.listdir(imgs_path)
len_ = len(imgs_path_list)
i = 0
for item in glob.glob(imgs_path + "/*.jpg"):
	# print(item)
	img0 = Image.open(os.path.join(imgs_path, item))
	img = img0.resize((img_w, img_h))
	img_np = np.asarray(img).astype(np.float32)
	mi = img_np.min()
	ma = img_np.max()
	img_np = (img_np - mi) / (ma - mi)
	img = img_np[:, :, :, np.newaxis]
	img_list.append(img)
	i += 1
	print(i, '/', len_)

imgs = np.concatenate(img_list, axis=3)  # values already in the range of 0-1

for i in range(3):
	pixels = imgs[:, :, i, :].ravel()
	means.append(np.mean(pixels))
	stdevs.append(np.std(pixels))

# BGR --> RGB ， CV读取的需要转换，PIL读取的不用转换
# means.reverse()
# stdevs.reverse()

print("MEAN = {}".format(means))
print("STD = {}".format(stdevs))

#---------- calculate eigenvector and eigenvalue of dataset ----------
# pixels = imgs[:, :, 2, :].ravel()  # flatten
# scaled_R = pixels - means[0]  # R
# pixels = imgs[:, :, 1, :].ravel()  # flatten
# scaled_G = pixels - means[1]  # G
# pixels = imgs[:, :, 0, :].ravel()  # flatten
# scaled_B = pixels - means[2]  # B
# cov=np.cov((scaled_R, scaled_G), scaled_B) #calculate covariance
# eig_val, eig_vec = np.linalg.eig(cov)
# print('eig_val = {}'.format(eig_val))
# print('eig_vec = {}'.format(eig_vec))

