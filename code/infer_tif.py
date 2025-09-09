import math
import os
import warnings

import albumentations as A
import numpy as np
import psutil
import torch
import torch.utils.data as D
from albumentations.pytorch import ToTensorV2
from osgeo import gdal
from tqdm import tqdm

from segmentation_models_pytorch import UnetPlusPlus

warnings.filterwarnings('ignore')
torch.backends.cudnn.enabled = True
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


#  计算可用内存
def memory_usage():
    mem_available = psutil.virtual_memory().available >> 20  # 可用内存
    mem_process = psutil.Process(os.getpid()).memory_info().rss >> 20  # 进程内存
    return mem_process, mem_available


# 计算分块数据
def get_block(width, height, bands):
    # return: 分块个数，每块行数，剩余行数
    p, a = memory_usage()
    bl = (a - 2000) / (width * height * bands >> 20)
    if bl > 3:
        block_size = 1
    else:
        block_size = math.ceil(bl) + 4

    bl_height = int(height / block_size)
    mod_height = height % block_size

    return block_size, bl_height, mod_height


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
        self.val_transform = A.Compose([
            # A.Normalize(),
            ToTensorV2()
        ])

    # 根据index获取小图片
    def __getitem__(self, index):
        image = self.image_list[index].transpose(1, 2, 0).astype(np.float32)
        transformed_data = self.val_transform(image=image)
        image = transformed_data['image']
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
    model = UnetPlusPlus(
        encoder_name="efficientnet-b0",
        encoder_weights=None,
        classes=2,
        in_channels=8,
        # decoder_attention_type='scse'
    )
    model.load_state_dict(torch.load(model_path))
    model.to(DEVICE)
    model.eval()
    return model





# 预测主函数
def predict(arr, area_perc, size, batch_size, model):
    arr = arr[:3,:,:]
    c, h, w = arr.shape
    RepetitiveLength = int((1 - math.sqrt(area_perc)) * size / 2)
    TifArray, RowOver, ColumnOver = TifCroppingArray(arr, RepetitiveLength, h, w, size)

    image_list = get_image_list(TifArray)
    test_loader = get_dataloader(image_list, batch_size=batch_size)
    result_list = []  # 结果列表
    with torch.no_grad():
        for image in tqdm(test_loader):

            image = image.to(DEVICE)
            image_flip1 = torch.flip(image, [2]).to(DEVICE)
            image_flip2 = torch.flip(image, [3]).to(DEVICE)
            output_list = []
            output1 = model(image).cpu().data.numpy()
            output2 = torch.flip(model(image_flip1), [2]).cpu().data.numpy()
            output3 = torch.flip(model(image_flip2), [3]).cpu().data.numpy()
            output_list.append(output1)
            output_list.append(output2)
            output_list.append(output3)
            output = np.mean(np.array(output_list), axis=0)
            for i in range(output.shape[0]):
                predict = np.argmax(output[i], axis=0).astype(np.uint8)
                result_list.append(predict * 255)
    result_shape = (arr.shape[1], arr.shape[2])
    result = Result(result_shape, TifArray, result_list, RepetitiveLength, RowOver, ColumnOver, np.uint8, size)
    return result


# 多线程预测需要程序位于main函数下
if __name__ == '__main__':
    image_path = r"F:\XJ\python_code\segmentation_baseline\data\2022\202301.tif"
    result_path = '../user_data/infer_result/remote_sensing_result/result1.tif'
    area_perc = 0.5
    size = 256
    batch_size = 16
    modelPath = '../user_data/model_data/seg_model_ibn.pth'
    model = torch.load(modelPath)
    ds = gdal.Open(image_path)
    # print(ds)
    width, height, bands = ds.RasterXSize, ds.RasterYSize, ds.RasterCount

    # 分块
    bl_size, bl_each, bl_mod = get_block(width, height, bands)
    # 提取分块区域位置(起点,行数)
    block_region = [(bs * bl_each, bl_each) for bs in range(bl_size)]
    if bl_mod != 0:
        block_region.append([bl_size * bl_each, bl_mod])

    # 输出结果保存
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(result_path, width, height, 1, gdal.GDT_Byte, options=["TILED=YES", "COMPRESS=LZW"])
    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())

    # 分块计算并保存
    for h_pos, h_num in block_region:
        print(f'start height pos:{h_pos}, end height pos:{h_pos + h_num - 1}')
        x = ds.ReadAsArray(0, h_pos, width, h_num)
        rst = predict(x, area_perc, size, batch_size, model)
        out_ds.GetRasterBand(1).WriteArray(rst, 0, h_pos)
        del x, rst

    out_ds.FlushCache()
    out_ds = None
