import numpy as np
import cv2
import glob


def to_png(array, path=None):
    arrays = np.load(array)
    # arrays[arrays == True] = 255
    # arrays[arrays != True] = 0
    # cv2.imwrite(path,arrays)
    print(arrays.shape)


npy_path = glob.glob(r"D:\01_python_code\segmentation_baseline\data\dataset\alt_masks/*.npy")
result_path = r"D:\01_python_code\segmentation_baseline\data\dataset\label/"
for i in npy_path:
    name = i.split("\\")[-1].replace("npy", "png")
    path = result_path + name
    to_png(i, path)
