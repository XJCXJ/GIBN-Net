# -*- coding: utf-8 -*-
"""
@Time    : 2023/2/19/019 19:11
@Author  : NDWX
@File    : split_image.py
@Software: PyCharm
"""
import os

from osgeo import gdal


def crop_images(input_image_path, input_label_path, output_path, crop_size=512, stride=256):
    # Open the input image and label files
    input_image = gdal.Open(input_image_path)
    input_label = gdal.Open(input_label_path)

    # Get the width and height of the input files
    image_width = input_image.RasterXSize
    image_height = input_image.RasterYSize

    # Calculate the number of crops to make in each dimension
    num_crops_x = (image_width - crop_size) // stride + 1
    num_crops_y = (image_height - crop_size) // stride + 1

    # Loop through each crop and save the images and labels
    for x in range(num_crops_x):
        for y in range(num_crops_y):
            # Calculate the starting and ending coordinates for the crop
            x_start = x * stride
            y_start = y * stride
            x_end = x_start + crop_size
            y_end = y_start + crop_size

            # Read the image and label data for the current crop
            image_data = input_image.ReadAsArray(x_start, y_start, crop_size, crop_size)
            label_data = input_label.ReadAsArray(x_start, y_start, crop_size, crop_size)

            # Create the output directory if it doesn't exist
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            if not os.path.exists(os.path.join(output_path, "image")):
                os.makedirs(os.path.join(output_path, "image"))
            if not os.path.exists(os.path.join(output_path, "label")):
                os.makedirs(os.path.join(output_path, "label"))

            # Save the image and label data for the current crop
            output_image_path = os.path.join(os.path.join(output_path, "image"), f"{x}_{y}.tif")
            output_label_path = os.path.join(os.path.join(output_path, "label"), f"{x}_{y}.tif")

            output_image = gdal.GetDriverByName('GTiff').Create(output_image_path, crop_size, crop_size,
                                                                input_image.RasterCount, gdal.GDT_Byte)
            output_image.SetGeoTransform(input_image.GetGeoTransform())
            output_image.SetProjection(input_image.GetProjection())
            # output_image.GetRasterBand(1).WriteArray(image_data)
            for i in range(input_image.RasterCount):
                output_image.GetRasterBand(i + 1).WriteArray(image_data[i])
            output_image.FlushCache()

            output_label = gdal.GetDriverByName('GTiff').Create(output_label_path, crop_size, crop_size, 1,
                                                                gdal.GDT_Byte)
            output_label.SetGeoTransform(input_label.GetGeoTransform())
            output_label.SetProjection(input_label.GetProjection())
            output_label.GetRasterBand(1).WriteArray(label_data)
            output_label.FlushCache()

    # Close the input files
    input_image = None
    input_label = None


if __name__ == '__main__':
    input_image_path = r"D:\jjw\Python\RS-Segmentation\cloud_detection\data\CSWV_S6\train_image\0.tif"
    input_label_path = r"D:\jjw\Python\RS-Segmentation\cloud_detection\data\CSWV_S6\train_mask\0.tif"
    output_path = r"D:\jjw\Python\RS-Segmentation\cloud_detection\data\dataset"
    crop_images(input_image_path, input_label_path, output_path, crop_size=512, stride=512)
