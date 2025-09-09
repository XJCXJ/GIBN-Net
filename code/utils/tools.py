# -*- coding: utf-8 -*-
"""
@Time    : 2022/11/28/028 13:46
@Author  : NDWX
@File    : tools.py
@Software: PyCharm
"""
import numpy as np
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import _LRScheduler


#  随机划分数据集
def split_dataset(dataset, random_seed):
    x_path, y_path = np.array(dataset[0]), np.array(dataset[1])
    folds = KFold(n_splits=5, shuffle=True, random_state=random_seed).split(range(len(x_path)),
                                                                            range(len(y_path)))
    (trn_idx, val_idx) = next(folds)
    train_dataset = [x_path[trn_idx], y_path[trn_idx]]
    val_dataset = [x_path[val_idx], y_path[val_idx]]
    return train_dataset, val_dataset


class PolynomialLRDecay(_LRScheduler):
    """Polynomial learning rate decay until step reach to max_decay_step

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_decay_steps: after this step, we stop decreasing learning rate
        end_learning_rate: scheduler stoping learning rate decay, value of learning rate must be this value
        power: The power of the polynomial.
    """

    def __init__(self, optimizer, max_decay_steps, end_learning_rate=0.0001, power=1.0):
        if max_decay_steps <= 1.:
            raise ValueError('max_decay_steps should be greater than 1.')
        self.max_decay_steps = max_decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.last_step = 0
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_step > self.max_decay_steps:
            return [self.end_learning_rate for _ in self.base_lrs]

        return [(base_lr - self.end_learning_rate) *
                ((1 - self.last_step / self.max_decay_steps) ** (self.power)) +
                self.end_learning_rate for base_lr in self.base_lrs]

    def step(self, step=None):
        if step is None:
            step = self.last_step + 1
        self.last_step = step if step != 0 else 1
        if self.last_step <= self.max_decay_steps:
            decay_lrs = [(base_lr - self.end_learning_rate) *
                         ((1 - self.last_step / self.max_decay_steps) ** (self.power)) +
                         self.end_learning_rate for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
                param_group['lr'] = lr
