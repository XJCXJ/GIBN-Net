"""变化检测数据裁剪"""
import argparse
import glob
import os
import random
import time
from osgeo import gdal
import numpy as np


#  创建文件夹
def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


#  读取栅格数据
def read(fileName, xoff=0, yoff=0, data_width=0, data_height=0):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    #  栅格矩阵的列数
    width = dataset.RasterXSize
    #  栅格矩阵的行数
    height = dataset.RasterYSize
    #  获取数据
    if (data_width == 0 and data_height == 0):
        data_width = width
        data_height = height
    data = dataset.ReadAsArray(xoff, yoff, data_width, data_height)
    #  获取仿射矩阵信息
    geotrans = dataset.GetGeoTransform()
    #  获取投影信息
    proj = dataset.GetProjection()
    return width, height, data, geotrans, proj


#  保存样本函数
def writeTiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
        # datatype = gdal.GDT_Byte
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
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset


#  选取正样本并保存
def select2write(dataA, label, geotrans, proj, savePath, name, False_sample, mask=(255, 128)):
    label = label.astype(np.uint8)
    # if (np.sum((dataA[1] == 0))) / (1024 * 1024) <= 0.8:
        # if 255 in label:
        #     if 0 not in label:
        #         if random.random() <= 0.2:
        #             writeTiff(dataA, geotrans, proj, os.path.join(savePath, "image/%d.png" % name))
        #             writeTiff(label, geotrans, proj, os.path.join(savePath, "label/%d.png" % name))
        #             name += 1
        #     else:
        #         writeTiff(dataA, geotrans, proj, os.path.join(savePath, "image/%d.png" % name))
        #         writeTiff(label, geotrans, proj, os.path.join(savePath, "label/%d.png" % name))
        #         name += 1
        # else:
        #     if random.random() <= 0.5:
    writeTiff(dataA, geotrans, proj, os.path.join(savePath, "image/%d.png" % name))
    writeTiff(label, geotrans, proj, os.path.join(savePath, "label/%d.png" % name))
    name += 1

    if label.shape != (1024, 1024):
         print(name)

    return name


#  裁剪函数
def Crop(width, height, imgA, label, geotrans, proj, SavePath, CropSize, RepetitionRate, False_sample):
    #  获取当前文件夹的文件个数len,并以len+1命名即将裁剪得到的图像
    new_name = len(os.listdir(os.path.join(SavePath, "image"))) + 1
    #  裁剪图片,重复率为RepetitionRate
    for i in range(int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
        for j in range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
            cropped_label = label[
                            int(i * CropSize * (1 - RepetitionRate)): int(
                                i * CropSize * (1 - RepetitionRate)) + CropSize,
                            int(j * CropSize * (1 - RepetitionRate)): int(
                                j * CropSize * (1 - RepetitionRate)) + CropSize]
            croppedA = imgA[:,
                       int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                       int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]

            new_name = select2write(croppedA, cropped_label, geotrans, proj, SavePath, new_name,
                                    False_sample)
    #  向前裁剪最后一列
    for i in range(int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
        cropped_label = label[
                        int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                        (width - CropSize): width]
        croppedA = imgA[:,
                   int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                   (width - CropSize): width]

        new_name = select2write(croppedA, cropped_label, geotrans, proj, SavePath, new_name, False_sample)
    #  向前裁剪最后一行
    for j in range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
        cropped_label = label[(height - CropSize): height,
                        int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]
        croppedA = imgA[:,
                   (height - CropSize): height,
                   int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]

        new_name = select2write(croppedA, cropped_label, geotrans, proj, SavePath, new_name, False_sample)
    #  裁剪右下角
    cropped_label = label[(height - CropSize): height,
                    (width - CropSize): width]
    croppedA = imgA[:,
               (height - CropSize): height,
               (width - CropSize): width]

    new_name = select2write(croppedA, cropped_label, geotrans, proj, SavePath, new_name, False_sample)


#  获取数据
def main(RasterA, Label):
    width, height, dataA, geotrans, proj = read(RasterA)
    label = read(Label)[2]
    # print(os.path.split(RasterA)[1], os.path.split(RasterB)[1],
    #       os.path.split(Label)[1], "栅格读取完毕,开始裁剪")
    return width, height, dataA, label, geotrans, proj


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--imageA', type=str, default=r"F:\XJ\python_code\segmentation_baseline\data\anshu_1024\image/",
                        help='ImageA path')
    # parser.add_argument('--imageB', type=str, default=r"", help='ImageB path')
    parser.add_argument('--label', type=str, default=r"F:\XJ\python_code\segmentation_baseline\data\anshu_1024\label/",
                        help='Label path')
    parser.add_argument('--output', type=str, default=r"F:\XJ\python_code\segmentation_baseline\data\anshu_1024\OUT/",
                        help='output path')
    parser.add_argument('--cropsize', type=int, default=1024)
    parser.add_argument('--RepetitionRate', type=float, default=0)
    parser.add_argument('--FalseSample', action='store_true', help='save false sample')
    opt = parser.parse_args()
    start = time.time()
    RasterA_list = glob.glob(os.path.join(opt.imageA, "*.tif"))
    Label_list = glob.glob(os.path.join(opt.label, "*.tif"))
    #  创建文件夹用于保存裁剪结果
    mkdir(os.path.join(opt.output, "image"))
    mkdir(os.path.join(opt.output, "label"))
    for i in range(len(RasterA_list)):
        print(RasterA_list[i], Label_list[i].replace("photo", "labels"))
        width, height, dataA, label, geotrans, proj = main(RasterA_list[i], Label_list[i].replace("photo", "labels"))
        dataA = dataA[:4, :, :]
        print(dataA.shape)

        Crop(width, height, dataA, label, geotrans, proj, opt.output, opt.cropsize, opt.RepetitionRate,
             opt.FalseSample)
    end = time.time()
    print("用时{}秒".format((end - start)))
