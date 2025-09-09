import os

import cv2
import numpy as np


def convert_to_single_channel_label(image_path, outpath):
    # 读取图片
    img = cv2.imread(image_path)

    # 定义颜色映射关系
    color_mapping = {
        (255, 255, 255): 1,  # 白色
        (0, 0, 255): 2,  # 红色
        (0, 255, 0): 3,  # 绿色
        (0, 255, 255): 4,  # 青色
        (255, 255, 0): 5,  # 黄色
        (255, 0, 0): 6  # 蓝色
    }

    # 将图像转换为单通道标签图
    single_channel_label = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for color, label in color_mapping.items():
        mask = np.all(img == np.array(color), axis=-1)
        single_channel_label[mask] = label

    # 保存结果
    cv2.imwrite(outpath, single_channel_label)


# 替换成你的图片路径
image_paths = r"D:\Data\pots\pots\test\label"
out_paths = r"D:\Data\pots\pots\test\label2"
image_list = os.listdir(image_paths)
for i in image_list:
    file = os.path.join(image_paths, i)
    convert_to_single_channel_label(file, os.path.join(out_paths, i))
