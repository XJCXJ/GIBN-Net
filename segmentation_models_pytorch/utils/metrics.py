from . import base
from . import functional as F
from ..base.modules import Activation
import torch


class IoU(base.Metric):
    __name__ = "iou_score"

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.iou(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )

# class IoU(base.Metric):
#     __name__ = "iou_score"
#
#     def __init__(self, num_classes=7, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
#         super(IoU, self).__init__(**kwargs)
#         self.eps = eps
#         self.threshold = threshold
#         self.activation = activation  # Assuming activation is already applied
#         self.ignore_channels = ignore_channels
#         self.num_classes = num_classes
#         self.intersection = torch.zeros(num_classes)
#         self.union = torch.zeros(num_classes)
#
#     def update(self, y_pr, y_gt):
#         y_pr = self.activation(y_pr)
#         for class_id in range(self.num_classes):
#             pred_mask = y_pr == class_id
#             true_mask = y_gt == class_id
#
#             intersection = torch.logical_and(true_mask, pred_mask).sum().item()
#             union = torch.logical_or(true_mask, pred_mask).sum().item()
#
#             self.intersection[class_id] += intersection
#             self.union[class_id] += union
#
#     def compute(self):
#         class_iou = torch.zeros(self.num_classes)
#         for class_id in range(self.num_classes):
#             iou = (self.intersection[class_id] + self.eps) / (self.union[class_id] + self.eps)
#             class_iou[class_id] = iou
#         mean_iou = torch.mean(class_iou)
#         return mean_iou

class Fscore(base.Metric):
    def __init__(self, beta=1, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.f_score(
            y_pr,
            y_gt,
            eps=self.eps,
            beta=self.beta,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Accuracy(base.Metric):
    def __init__(self, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.accuracy(
            y_pr,
            y_gt,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Recall(base.Metric):
    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.recall(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Precision(base.Metric):
    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.precision(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )
