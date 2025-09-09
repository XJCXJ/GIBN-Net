# -*- coding: utf-8 -*-
"""
@Time    : 2022/11/22/022 13:53
@Author  : NDWX
@File    : test.py
@Software: PyCharm
"""
import glob

import torch

import segmentation_models_pytorch as smp
from utils.dataset import build_test_dataloader

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    model_name = "IBN_1024_3_res18_ttff"
    # 测试数据集
    test_dataset = [sorted(glob.glob("../data/Anshu_data/test/image/*.png")),
                    sorted(glob.glob("../data/Anshu_data/test/label/*.png"))]
    # 模型路径
    model = torch.load('../user_data/model_data/seg_model_{}.pth'.format(model_name))

    # loss = smp.utils.losses.DiceLoss()
    loss = torch.nn.CrossEntropyLoss()
    loss.__name__ = "CrossEntropyLoss"

    metrics = [
        smp.utils.metrics.Accuracy(activation='argmax2d'),
        smp.utils.metrics.Precision(activation='argmax2d'),
        smp.utils.metrics.Recall(activation='argmax2d'),
        smp.utils.metrics.IoU(activation='argmax2d'),
        smp.utils.metrics.Fscore(activation='argmax2d')
    ]

    test_loader = build_test_dataloader(test_dataset)

    test_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    test_epoch.infer_vis(test_loader, evaluate=True, save=True, save_dir='../user_data/infer_result/anshu', suffix=".png")

