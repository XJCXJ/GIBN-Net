# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

plt.rc('font', family='Times New Roman')  # 设置图表为新罗马字体
plt.rcParams["axes.labelweight"] = "bold"  # 坐标轴字体加粗
model_name = "Unet++_1024"
a = np.load("../user_data/figure_data/figure_{}.npy".format(model_name))

a_loss, a_iou, a_P, a_R, a_F1 = a

epochs = range(1, len(a_loss) + 1)

plt.plot(range(1, len(a_loss) + 1), a_loss, color='#9D2EC5', label="loss")
plt.plot(range(1, len(a_loss) + 1), a_iou, color='#70DB93', label="IoU")
plt.plot(range(1, len(a_loss) + 1), a_P, color='#A67D3D', label="P")
plt.plot(range(1, len(a_loss) + 1), a_R, color='#FF7F00', label="R")
plt.plot(range(1, len(a_loss) + 1), a_F1, color='#238E23', label="F1")



plt.xlabel("Epochs")  # 横坐标名字
plt.ylabel("Loss value")  # 纵坐标名字
plt.legend(loc="best")  # 图例
plt.savefig("../user_data/figure_data/loss_epoch.png", dpi=300)
plt.show()
