# -*- coding: utf-8 -*-
"""
@Time    : 2023/3/29/029 14:41
@Author  : NDWX
@File    : train.py
@Software: PyCharm
"""
import glob
import os
import random
import warnings

import numpy as np
import torch

import segmentation_models_pytorch as smp
import utils
from nets.network import Unet
from utils.dataset import build_dataloader
from utils.tools import PolynomialLRDecay
from utils.metrics import cal_iou
from nets.ibn.models import ResNetFPN

warnings.filterwarnings('ignore')
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# 固定随机种子
def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# 加载模型
# def load_model(DEVICE, classes):
#     model = smp.FPN(
#         encoder_name="resnet50",#efficientnet-b0
#         classes=classes,
#         in_channels=channels,
#         # decoder_attention_type='scse',
#         # seg_ensemble='ecam',
#         # activation='softmax'
#     )
#     model.to(DEVICE)
#     return model


#
# 使用自己的模型
def load_model(DEVICE, classes):
    model = ResNetFPN(num_classes=classes)
    model.to(DEVICE)
    return model


if __name__ == '__main__':
    random_seed = 1007
    num_epochs = 50
    batch_size = 2
    channels = 3
    lr = 1e-4
    setup_seed(random_seed)
    model_name = "IBN_1024_3_preN"
    train_dataset = [sorted(glob.glob("../data/Anshu_data/train/image/*.png")),
                     sorted(glob.glob("../data/Anshu_data/train/label/*.png"))]
    val_dataset = [sorted(glob.glob("../data/Anshu_data/val/image/*.png")),
                   sorted(glob.glob("../data/Anshu_data/val/label/*.png"))]

    # train_dataset, val_dataset = split_dataset(dataset, random_seed)
    train_loader, valid_loader = build_dataloader(train_dataset, val_dataset, int(batch_size))

    model_save_path = "../user_data/model_data/seg_model_{}.pth".format(model_name)
    model = load_model(DEVICE, classes=2)

    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=lr)])
    scheduler = PolynomialLRDecay(optimizer, num_epochs, 1e-5)

    loss = torch.nn.CrossEntropyLoss()
    loss.__name__ = "CrossEntropyLoss"

    metrics = [
        smp.utils.metrics.Precision(activation='argmax2d'),
        smp.utils.metrics.Recall(activation='argmax2d'),
        smp.utils.metrics.IoU(activation='argmax2d'),
        smp.utils.metrics.Fscore(activation='argmax2d'),
    ]

    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )
    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    max_score = 0
    train_loss, val_mIoU, val_precision, val_recall, val_F1 = [], [], [], [], []

    for i in range(num_epochs):

        print('\nEpoch: {}'.format(i + 1))

        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        # print(valid_logs)
        scheduler.step()

        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            print('max_score', max_score)
            torch.save(model, '../user_data/model_data/seg_model_{}.pth'.format(model_name))
            print('Model saved!')
        #
        train_loss.append(train_logs['CrossEntropyLoss']), val_mIoU.append(valid_logs['iou_score']), val_precision.append(
            valid_logs['precision']), val_recall.append(valid_logs['recall']), val_F1.append(valid_logs['fscore'])

    np.save("../user_data/figure_data/figure_{}.npy".format(model_name), [train_loss, val_mIoU, val_precision, val_recall, val_F1])

