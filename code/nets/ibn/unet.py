import glob
import os
import random
import warnings

import numpy as np
import torch

import segmentation_models_pytorch as smp


warnings.filterwarnings('ignore')
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def load_model(DEVICE, classes):
    model = smp.FPN(
        encoder_name="resnet18",
        classes=classes,
        in_channels=3,
        )
    model.to(DEVICE)
    return model

if __name__ == '__main__':
    model = smp.FPN(encoder_name="resnet18",
        classes=2,
        in_channels=3,
        upsampling=4,
        )
    x1 = torch.randn(4, 3, 256, 256)
    output = model(x1)
    print(output.size())