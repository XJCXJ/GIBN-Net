import os
import cv2
import numpy as np


def read_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        images.append(image)
    return images


def calculate_tp_fp_fn(labels, predictions, num_classes=6):
    tp_per_class = np.zeros(num_classes)
    fp_per_class = np.zeros(num_classes)
    fn_per_class = np.zeros(num_classes)

    for i in range(1, num_classes + 1):
        # Create binary masks for each class
        label_i = (labels == i)
        prediction_i = (predictions == i)

        # Calculate true positive (TP), false positive (FP), and false negative (FN)
        tp = np.sum(np.logical_and(label_i, prediction_i))
        fp = np.sum(np.logical_and(np.logical_not(label_i), prediction_i))
        fn = np.sum(np.logical_and(label_i, np.logical_not(prediction_i)))

        # Accumulate TP, FP, and FN for each class
        tp_per_class[i - 1] += tp
        fp_per_class[i - 1] += fp
        fn_per_class[i - 1] += fn

    return tp_per_class, fp_per_class, fn_per_class


# Example folder paths
label_folder = r"D:\01_python_code\segmentation_baseline\data\vaihingen_dataset\test\label"
prediction_folder= r"D:\01_python_code\segmentation_baseline\user_data\infer_result"


# Read images from folders
label_images = read_images_from_folder(label_folder)
prediction_images = read_images_from_folder(prediction_folder)

# Assuming the images contain class labels (1 to 6)
total_tp_per_class = np.zeros(6)
total_fp_per_class = np.zeros(6)
total_fn_per_class = np.zeros(6)

for label_image, prediction_image in zip(label_images, prediction_images):
    tp_per_class, fp_per_class, fn_per_class = calculate_tp_fp_fn(label_image, prediction_image)

    total_tp_per_class += tp_per_class
    total_fp_per_class += fp_per_class
    total_fn_per_class += fn_per_class



def calculate_iou_from_tp_fp_fn(tp_per_class, fp_per_class, fn_per_class):
    iou_per_class = np.zeros_like(tp_per_class, dtype=float)

    for i in range(len(tp_per_class)):
        # Calculate IoU for each class
        denominator = tp_per_class[i] + fp_per_class[i] + fn_per_class[i]
        iou_per_class[i] = tp_per_class[i] / denominator if denominator > 0 else 0

    return iou_per_class

# Assuming you've calculated total_tp_per_class, total_fp_per_class, and total_fn_per_class
iou_per_class = calculate_iou_from_tp_fp_fn(total_tp_per_class, total_fp_per_class, total_fn_per_class)

# Print results for each class
for i in range(6):
    print(f'Class {i+1}: IoU = {iou_per_class[i]:.4f}')

