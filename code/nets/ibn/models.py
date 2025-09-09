import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from .resibn import *
from .decoder import FPNDecoder
from segmentation_models_pytorch.base import SegmentationHead


class ResNetFPN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = resnet18_ibn_b(pretrained=False)  # 使用预训练的 resnet50_ibn_b 作为编码器
        self.decoder = FPNDecoder(encoder_channels=[64, 128, 256, 512],  # 根据 ResNet18,34 的结构设置[64, 128, 256, 512],
                                  encoder_depth=4,  # 根据resnet50，101，设置成[256, 512, 1024,2048]
                                  pyramid_channels=256,
                                  segmentation_channels=128,
                                  dropout=0.2,
                                  merge_policy="add")

        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=num_classes,
            activation=None,  # 如果你希望在模型内部应用激活函数，可以在这里指定激活函数，否则设为 None
            kernel_size=1,
            upsampling=2,
        )
        # self.conv2 = torch.nn.Conv2d(2, 1, kernel_size=(1, 1))

    def forward(self, x):

        features = self.encoder(x)
        # print(len(features))
        x = self.decoder(*features)
        x = self.segmentation_head(x)
        # out = torch.sigmoid(conv2(x)).cpu().data.numpy()
        # x_visualize = out.squeeze()
        # x_visualize = (
        #         ((x_visualize - np.min(x_visualize)) / (np.max(x_visualize) - np.min(x_visualize))) * 255
        # ).astype(np.uint8)
        # x_visualize = cv2.applyColorMap(x_visualize, cv2.COLORMAP_JET)
        # save_path = r"F:\XJ\python_code\segmentation_baseline\user_data\infer_result\CAM/"
        # i = len(os.listdir((save_path)))
        # cv2.imwrite(save_path + str(i) + ".png", x_visualize)
        return x


if __name__ == '__main__':
    model = ResNetFPN(num_classes=2)  # 实例化 FPN 模型，指定编码器名称和类别数
    x1 = torch.randn(4, 4, 256, 256)  # 创建一个示例输入张量
    output = model(x1)  # 使用模型进行前向传播
    print(output.size())  # 打印输出的尺寸
