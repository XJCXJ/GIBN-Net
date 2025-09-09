# -*- coding: utf-8 -*-
"""
@Time    : 2023/4/3
@Author  : wdbd
@File    : infer_tif.py
@Software: PyCharm
"""

import glob
import os
import time
import math
import numpy as np
import torch
from tqdm import tqdm
from osgeo import gdal
from torchvision import transforms as T
import warnings
import torch.utils.data as D

# from Sea_ice_ConvLSTM_loss import *

warnings.filterwarnings('ignore')
torch.backends.cudnn.enabled = True
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


#  读取tif数据集
def readTif(fileName, xoff=0, yoff=0, data_width=0, data_height=0):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    #  栅格矩阵的列数
    width = dataset.RasterXSize
    #  栅格矩阵的行数
    height = dataset.RasterYSize
    #  波段数
    bands = dataset.RasterCount
    #  获取数据
    if (data_width == 0 and data_height == 0):
        data_width = width
        data_height = height
    data = dataset.ReadAsArray(xoff, yoff, data_width, data_height)
    #  获取仿射矩阵信息
    geotrans = dataset.GetGeoTransform()
    #  获取投影信息
    proj = dataset.GetProjection()
    return width, height, bands, data, geotrans, proj


#  保存tif文件函数
def writeTiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape

    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype,
                            options=["TILED=YES", "COMPRESS=LZW"])
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i, :, :])
    del dataset


#  tif裁剪（tif像素数据，裁剪边长）
def TifCroppingArray(im_data, SideLength, im_height, im_width, size):
    img = im_data
    TifArrayReturn = []
    #  列上图像块数目
    if (len(img.shape) == 2):
        #  列上图像块数目
        ColumnNum = int((img.shape[0] - SideLength * 2) / (size - SideLength * 2))
        #  行上图像块数目
        RowNum = int((img.shape[1] - SideLength * 2) / (size - SideLength * 2))
    else:
        #  列上图像块数目
        ColumnNum = int((im_height - SideLength * 2) / (size - SideLength * 2))
        #  行上图像块数目
        RowNum = int((im_width - SideLength * 2) / (size - SideLength * 2))
    for i in range(ColumnNum):
        TifArray = []
        for j in range(RowNum):
            if (len(img.shape) == 2):
                cropped = img[i * (size - SideLength * 2): i * (size - SideLength * 2) + size,
                          j * (size - SideLength * 2): j * (size - SideLength * 2) + size]
            else:
                cropped = img[:,
                          i * (size - SideLength * 2): i * (size - SideLength * 2) + size,
                          j * (size - SideLength * 2): j * (size - SideLength * 2) + size]
            TifArray.append(cropped)
        TifArrayReturn.append(TifArray)
    #  考虑到行列会有剩余的情况，向前裁剪一行和一列
    #  向前裁剪最后一列
    for i in range(ColumnNum):
        if (len(img.shape) == 2):
            cropped = img[i * (size - SideLength * 2): i * (size - SideLength * 2) + size,
                      (img.shape[1] - size): img.shape[1]]
        else:
            cropped = img[:,
                      i * (size - SideLength * 2): i * (size - SideLength * 2) + size,
                      (im_width - size): im_width]
        TifArrayReturn[i].append(cropped)
    #  向前裁剪最后一行
    TifArray = []
    for j in range(RowNum):
        if (len(img.shape) == 2):
            cropped = img[(img.shape[0] - size): img.shape[0],
                      j * (size - SideLength * 2): j * (size - SideLength * 2) + size]
        else:
            cropped = img[:,
                      (im_height - size): im_height,
                      j * (size - SideLength * 2): j * (size - SideLength * 2) + size]
        TifArray.append(cropped)
    #  向前裁剪右下角
    if (len(img.shape) == 2):
        cropped = img[(img.shape[0] - size): img.shape[0],
                  (img.shape[1] - size): img.shape[1]]
    else:
        cropped = img[:,
                  (im_height - size): im_height,
                  (im_width - size): im_width]
    TifArray.append(cropped)
    TifArrayReturn.append(TifArray)
    if (len(img.shape) == 2):
        #  列上的剩余数
        ColumnOver = (img.shape[0] - SideLength * 2) % (size - SideLength * 2) + SideLength
        #  行上的剩余数
        RowOver = (img.shape[1] - SideLength * 2) % (size - SideLength * 2) + SideLength
    else:
        #  列上的剩余数
        ColumnOver = (img.shape[1] - SideLength * 2) % (size - SideLength * 2) + SideLength
        #  行上的剩余数h
        RowOver = (img.shape[2] - SideLength * 2) % (size - SideLength * 2) + SideLength
    return TifArrayReturn, RowOver, ColumnOver


#  获得结果矩阵
def Result(shape, TifArray, npyfile, RepetitiveLength, RowOver, ColumnOver, format, size):
    result = np.zeros(shape, format)
    #  j来标记行数
    j = 0
    for i, item in enumerate(npyfile):
        img = item
        img = img.reshape(size, size)
        #  最左侧一列特殊考虑，左边的边缘要拼接进去
        if (i % len(TifArray[0]) == 0):
            #  第一行的要再特殊考虑，上边的边缘要考虑进去
            if (j == 0):
                result[0: size - RepetitiveLength, 0: size - RepetitiveLength] = img[0: size - RepetitiveLength,
                                                                                 0: size - RepetitiveLength]
            #  最后一行的要再特殊考虑，下边的边缘要考虑进去
            elif (j == len(TifArray) - 1):
                #  后来修改的
                result[shape[0] - ColumnOver - RepetitiveLength: shape[0], 0: size - RepetitiveLength] = img[
                                                                                                         size - ColumnOver - RepetitiveLength: size,
                                                                                                         0: size - RepetitiveLength]
            else:
                result[j * (size - 2 * RepetitiveLength) + RepetitiveLength: (j + 1) * (
                        size - 2 * RepetitiveLength) + RepetitiveLength,
                0:size - RepetitiveLength] = img[RepetitiveLength: size - RepetitiveLength, 0: size - RepetitiveLength]
        #  最右侧一列特殊考虑，右边的边缘要拼接进去
        elif (i % len(TifArray[0]) == len(TifArray[0]) - 1):
            #  第一行的要再特殊考虑，上边的边缘要考虑进去
            if (j == 0):
                result[0: size - RepetitiveLength, shape[1] - RowOver: shape[1]] = img[0: size - RepetitiveLength,
                                                                                   size - RowOver: size]
            #  最后一行的要再特殊考虑，下边的边缘要考虑进去
            elif (j == len(TifArray) - 1):
                result[shape[0] - ColumnOver: shape[0], shape[1] - RowOver: shape[1]] = img[size - ColumnOver: size,
                                                                                        size - RowOver: size]
            else:
                result[j * (size - 2 * RepetitiveLength) + RepetitiveLength: (j + 1) * (
                        size - 2 * RepetitiveLength) + RepetitiveLength,
                shape[1] - RowOver: shape[1]] = img[RepetitiveLength: size - RepetitiveLength, size - RowOver: size]
            #  走完每一行的最右侧，行数+1
            j = j + 1
        #  不是最左侧也不是最右侧的情况
        else:
            #  第一行的要特殊考虑，上边的边缘要考虑进去
            if (j == 0):
                result[0: size - RepetitiveLength,
                (i - j * len(TifArray[0])) * (size - 2 * RepetitiveLength) + RepetitiveLength: (i - j * len(
                    TifArray[0]) + 1) * (size - 2 * RepetitiveLength) + RepetitiveLength
                ] = img[0: size - RepetitiveLength, RepetitiveLength: size - RepetitiveLength]
            #  最后一行的要特殊考虑，下边的边缘要考虑进去
            if (j == len(TifArray) - 1):
                result[shape[0] - ColumnOver: shape[0],
                (i - j * len(TifArray[0])) * (size - 2 * RepetitiveLength) + RepetitiveLength: (i - j * len(
                    TifArray[0]) + 1) * (size - 2 * RepetitiveLength) + RepetitiveLength
                ] = img[size - ColumnOver: size, RepetitiveLength: size - RepetitiveLength]
            else:
                result[j * (size - 2 * RepetitiveLength) + RepetitiveLength: (j + 1) * (
                        size - 2 * RepetitiveLength) + RepetitiveLength,
                (i - j * len(TifArray[0])) * (size - 2 * RepetitiveLength) + RepetitiveLength: (i - j * len(
                    TifArray[0]) + 1) * (size - 2 * RepetitiveLength) + RepetitiveLength,
                ] = img[RepetitiveLength: size - RepetitiveLength, RepetitiveLength: size - RepetitiveLength]
    return result


# 返回小图像列表
def get_image_list(tif_array):
    image_List = []
    for i in range(len(tif_array)):
        for j in range(len(tif_array[i])):
            image_List.append(tif_array[i][j])
    return image_List


# 定义测试Dataset类
class TestDataset(D.Dataset):
    # __init__()来存储数据路径和模式
    def __init__(self, image_list):
        self.image_list = image_list
        self.test_transform = T.Compose([
            T.ToTensor(),
        ])

    # 根据index获取小图片
    def __getitem__(self, index):
        image = self.image_list[index]
        image = self.test_transform(image.transpose(1, 2, 0))
        return image

    def __len__(self):
        return len(self.image_list)


# 定义测试dataloader
def get_dataloader(image_list, batch_size):
    dataset = TestDataset(image_list)
    dataloader = D.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)
    return dataloader


# 加载模型
def load_model(DEVICE, model_path):
    model = torch.load(model_path)
    model.to(DEVICE)
    model.eval()
    return model


# 多线程预测需要程序位于main函数下
if __name__ == '__main__':
    output_dir = r'F:\XJ\python_code\segmentation_baseline\user_data\infer_result\11'#结果保存文件夹
    test_image_list = glob.glob(r"F:\XJ\python_code\segmentation_baseline\data\2022/"+"*.tif")  # 获取待预测文件夹内全部遥感影像
    print(test_image_list)
    model_name = "IBN_1024_3"
    modelPath = '../user_data/model_data/seg_model_{}.pth'.format(model_name) #模型文件保存路径
    area_perc = 0.25
    size = 1024
    batch_size = 4
    RepetitiveLength = int((1 - math.sqrt(area_perc)) * size / 2)

    model = load_model(DEVICE, modelPath)


    for tif_image in test_image_list:
        stime = time.time()
        fileAName = os.path.split(tif_image)[1]
        width, height, bands, data, geotrans, proj = readTif(tif_image)
        data = data[:3,:,:]
        TifArray, RowOver, ColumnOver = TifCroppingArray(data, RepetitiveLength, height, width, size)
        image_list = get_image_list(TifArray)
        test_loader = get_dataloader(image_list, batch_size=batch_size)
        result_list = []  # 结果列表
        with torch.no_grad():
            for images in tqdm(test_loader):
                image = images.to(DEVICE)
                outputs = model(image).cpu().data.numpy()
                for i in range(outputs.shape[0]):
                    # pred = outputs[i]
                    # ret, prediction = torch.max(pred, 1)
                    # prediction = prediction.astype(np.uint8)
                    # prediction = prediction.reshape((256, 256))
                    # result_list.append(prediction)
                    predict = np.argmax(outputs[i], axis=0).astype(np.uint8)
                    result_list.append(predict * 255)
        result_shape = (data.shape[1], data.shape[2])
        result = Result(result_shape, TifArray, result_list, RepetitiveLength, RowOver, ColumnOver, np.uint8, size)
        result_path = os.path.join(output_dir, fileAName)
        writeTiff(result.astype(np.uint8), geotrans, proj, result_path)
