#!usr/bin/env python  
# -*- coding:utf-8 _*-
"""
@version: python3.6
"""
import sys
import os

from PIL import Image
from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(self, root=None, transform=None, target_transform=None, to=None):
        self.root = root
        data_list = os.listdir(root)
        self.env = []
        for i in data_list:
            # print(i)
            if os.path.splitext(i)[1] == '.txt':
                # print(i)
                self.env.append(root + '/' + i)

        self.len = len(self.env) - 1

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        img_path, label = open(self.env[index]).readlines()[0].split(',')
        img_path = self.root + '/' + img_path
        try:
            img = Image.open(img_path)
        except:
            print(img_path)
            print('Corrupted image for %d' % index)
            return self[index + 1]

        if self.transform is not None:
            if img.layers == 1:
                print(img_path)
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)
        return (img, int(label))


class TestDataset(Dataset):
    def __init__(self, root=None, transform=None, target_transform=None, to=None):
        if '.txt' in root:
            self.env = list(open(root))
        else:
            self.env = root

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        self.len = len(self.env) - 1

        self.transform = transform
        self.target_transform = target_transform
        self.to = to

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        img_path, label = self.env[index].strip().split(',')

        try:
            img = Image.open(img_path)
        except:
            print(img_path)
            print('Corrupted image for %d' % index)
            return self[index + 1]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return (img, int(label))


class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        # padding
        ratio = self.size[0] / self.size[1]
        w, h = img.size
        if w / h < ratio:
            t = int(h * ratio)
            w_padding = (t - w) // 2
            img = img.crop((-w_padding, 0, w + w_padding, h))
        else:
            t = int(w / ratio)
            h_padding = (t - h) // 2
            img = img.crop((0, -h_padding, w, h + h_padding))

        # img.show()
        # resize
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


if __name__ == '__main__':
    import torch.utils.data as data
    import os
    from torchvision import transforms

    data_path = '/data/multi_task/Ubuntu_Code/garbage_classify/huawei-garbage-master/Data/garbage_classify'
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = Dataset(root=data_path, transform=transform)
    train_loader = data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=2,
                                   pin_memory=True)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        print(inputs.size())
        print(targets)
        # img = inputs.squeeze().permute(1, 2, 0)
        # plt.imshow(img)
        # plt.show()
