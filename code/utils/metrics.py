# -*- coding: utf-8 -*-
"""
@Time    : 2022/11/28/028 13:45
@Author  : NDWX
@File    : metrics.py
@Software: PyCharm
"""
import numpy as np
import torch

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


#  验证集不需要梯度计算,加速和节省gpu空间
@torch.no_grad()
# 计算验证集Iou
# def cal_val_iou(model, loader):
#     val_iou = []
#     model.eval()
#     for image, target in loader:
#         image, target = image.to(DEVICE), target.to(DEVICE)
#         output = model(image)
#         output = output.argmax(1)
#         iou = cal_iou(output, target)
#         val_iou.append(iou)
#     return val_iou
#
#
# # 计算IoU
def cal_iou(pred, mask, c=6):
    iou_result = []
    for idx in range(c):
        p = (mask == idx).int().reshape(-1)
        t = (pred == idx).int().reshape(-1)
        uion = p.sum() + t.sum()
        overlap = (p*t).sum()
        #  0.0001防止除零
        iou = 2*overlap/(uion + 0.0001)
        iou_result.append(iou.abs().data.cpu().numpy())
    return np.stack(iou_result)

# def cal_iou(y_true, y_pred, num_classes):
#     iou_scores = []
#     for class_id in range(1, num_classes + 1):
#         true_mask = y_true == class_id
#         pred_mask = y_pred == class_id
#
#         intersection = np.logical_and(true_mask, pred_mask)
#         union = np.logical_or(true_mask, pred_mask)
#
#         iou = np.sum(intersection) / np.sum(union)
#         iou_scores.append(iou)
#
#     mean_iou = np.mean(iou_scores)
#     return mean_iou

def calculate_dice_coefficient(y_true, y_pred, num_classes):
    dice_scores = []
    for class_id in range(1, num_classes + 1):
        true_mask = y_true == class_id
        pred_mask = y_pred == class_id

        intersection = np.sum(np.logical_and(true_mask, pred_mask))
        union = np.sum(true_mask) + np.sum(pred_mask)

        dice = (2.0 * intersection) / union
        dice_scores.append(dice)

    mean_dice = np.mean(dice_scores)
    return mean_dice