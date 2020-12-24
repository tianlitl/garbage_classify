#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from .Resnet import resnet50
from .dataset import Dataset
from .metrics import accuracy
from .model import PMG
from .utils import print_msg


class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        ratio = 1
        w, h = img.size
        if w / h < ratio:
            t = int(h * ratio)
            w_padding = (t - w) // 2
            img = img.crop((-w_padding, 0, w + w_padding, h))
        else:
            t = int(w / ratio)
            h_padding = (t - h) // 2
            img = img.crop((0, -h_padding, w, h + h_padding))

        img = img.resize(self.size, self.interpolation)

        return img


def _eval(model, dataloader):
    model.eval()
    torch.set_grad_enabled(False)

    correct = 0
    total = 0
    test_loss = 0
    correct_com = 0

    all_targ = torch.tensor([]).to(dtype=torch.int64).cuda()
    all_pred = torch.tensor([]).to(dtype=torch.int64).cuda()

    for test_data in dataloader:
        X, y = test_data
        X, y = X.cuda(), y.cuda()

        # y_pred = model(X)
        output_1, output_2, output_3, output_concat = model(X)
        outputs_com = output_1 + output_2 + output_3 + output_concat
        criterion = torch.nn.CrossEntropyLoss().cuda()
        loss = criterion(output_concat, y)

        test_loss += loss.item()
        _, predicted = torch.max(output_concat.data, 1)
        _, predicted_com = torch.max(outputs_com.data, 1)
        total += y.size(0)
        # correct += predicted.eq(y.data).cpu().sum()
        # correct_com += predicted_com.eq(y.data).cpu().sum()

        all_pred = torch.cat((all_pred, predicted_com))
        all_targ = torch.cat((all_targ, y.to(torch.int64)))
        # total += y.size(0)
        correct += accuracy(predicted_com, y) * y.size(0)

    acc = round(correct / total, 4)
    model.train()
    torch.set_grad_enabled(True)
    return acc, all_pred.cpu().numpy()


if __name__ == '__main__':
    # creat train dataset
    data_path = '/data/multi_task/Ubuntu_Code/garbage_classify/huawei-garbage-master/Data'
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    transform = transforms.Compose([Resize((448, 448)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=mean, std=std)])
    train_dataset = Dataset(root=data_path, transform=transform)  # config.py
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              pin_memory=True)
    test_dataset = Dataset(root=data_path, transform=transform)  # config.py
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             pin_memory=True)

    # creat model
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    net = resnet50(pretrained=True)
    model = PMG(net, 512, 43)
    # print(model)
    model = torch.nn.DataParallel(model)
    model = model.cuda()

    model = model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    EPOCH = 150
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=0.01,
                                momentum=0.9,
                                nesterov=True,
                                weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        # 当训练epoch达到milestones值时,初始学习率乘以gamma得到新的学习率;
                                                        milestones=[50, 100],
                                                        gamma=0.5)
    resume = None
    if resume:
        # load checkpoint
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(resume), 'Error: no checkpoint directory found!'
        checkpoint = os.path.dirname(resume)
        checkpoint = torch.load(resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.module.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    max_acc = 0
    for epoch in range(1, EPOCH + 1):
        # learning rate update
        epoch_loss = 0
        correct = 0
        total = 0
        progress = tqdm(enumerate(train_loader))  # 进度条
        for step, train_data in progress:
            X, y = train_data  # X.dtype is torch.float32, y.dtype is torch.int64
            X, y = X.cuda(), y.float().cuda()

            # forward
            y_pred = model(X)[-1]
            print(f'label:{y} | prediction:{torch.argmax(y_pred, dim=1)}')

            loss = criterion(y_pred, y.long())

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if lr_scheduler:
                lr_scheduler.step()
                if epoch in lr_scheduler.milestones:
                    print_msg('Learning rate decayed to {}'.format(lr_scheduler.get_lr()[0]))

            # metrics
            epoch_loss += loss.item()
            total += y.size(0)
            correct += accuracy(torch.argmax(y_pred, dim=1), y) * y.size(0)
            avg_loss = epoch_loss / (step + 1)
            train_acc = correct / total

            progress.set_description(
                'Epoch: {}/{}, loss: {:.6f}, acc: {:.4f}'.format(epoch, EPOCH, avg_loss, train_acc))

            test_acc, all_pred = _eval(model, test_loader)
            if test_acc > max_acc:
                max_acc = test_acc

                state = {
                    'fold': 0,
                    'epoch': epoch + 1,
                    'state_dict': model.module.state_dict(),
                    'train_acc': train_acc,
                    'acc': test_acc,
                    'best_acc': max_acc,
                    'optimizer': optimizer.state_dict(),
                }
                result_path = '/data/multi_task/Ubuntu_Code/garbage_classify/result'
                torch.save(state, os.path.join(result_path, 'best_acc.pth'))
                print_msg(f'Epoch{epoch} | test_acc:{test_acc}')
