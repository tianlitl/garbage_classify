# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=43, smoothing=0.01):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.classes = classes
        self.criterion = nn.KLDivLoss(reduction='sum')

    def forward(self, pred, label):
        """
        Args:
            pred: [B, 43]
            label: [B]
        Returns:
            loss: int
        """
        assert pred.size(1) == self.classes
        pred = F.log_softmax(pred, dim=1)
        smooth_label = self.smooth_one_hot(pred, label)
        return self.criterion(pred, smooth_label)

    @torch.no_grad()
    def smooth_one_hot(self, pred, label):
        assert 0 <= self.smoothing < 1, "the smoothing value must belong to [0, 1)"
        confidence = 1.0 - self.smoothing

        smooth_label = torch.empty(size=pred.shape, device=pred.device)
        smooth_label.fill_(self.smoothing / (self.classes - 1))
        smooth_label.scatter_(1, label.unsqueeze(1), confidence)
        return smooth_label


if __name__ == '__main__':
    # torch.cuda.set_device(3)

    pred = torch.randn([4, 1, 25, 25]).cuda()
    label = torch.zeros([4, 25, 25]).cuda()
    label[3:8, :, :] = 1
    label[10:15, :, :] = -1

    model = LabelSmoothingLoss().cuda()
    loss = model(pred, label)
    print(loss)