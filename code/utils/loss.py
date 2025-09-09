# -*- coding: utf-8 -*-
"""
@Time    : 2022/11/28/028 13:41
@Author  : NDWX
@File    : losses.py
@Software: PyCharm
"""
import torch.nn as nn
from segmentation_models_pytorch.losses import DiceLoss


# 混合loss
class hyperloss(nn.Module):
    __name__ = 'hyperloss'

    def __init__(self):
        super(hyperloss, self).__init__()
        self.CELoss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.DiceLoss_fn = DiceLoss(mode='multiclass')

    def forward(self, pred, mask):
        loss_ce = self.CELoss_fn(pred, mask)
        loss_dice = self.DiceLoss_fn(pred, mask)
        loss = loss_ce + loss_dice
        return loss
