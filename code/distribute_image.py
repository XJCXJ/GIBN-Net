import os
import random
import shutil


#  创建文件夹
def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


image_root = r"F:\XJ\python_code\segmentation_baseline\data\dataset3\OUT/"
image_A = os.path.join(image_root, "image")
image_L = os.path.join(image_root, "label")

out_root = r"F:\XJ\python_code\segmentation_baseline\data\dataset3/"
result_t = os.path.join(out_root, "train")
result_v = os.path.join(out_root, "val")
result_test = os.path.join(out_root, "test")



mkdir(result_t)
mkdir(os.path.join(result_t,"image"))

mkdir(os.path.join(result_t,"label"))
mkdir(result_v)
mkdir(os.path.join(result_v,"image"))

mkdir(os.path.join(result_v,"label"))
mkdir(result_test)
mkdir(os.path.join(result_test,"image"))

mkdir(os.path.join(result_test,"label"))


result_train_num = os.path.join(result_t, "image")
result_val_num = os.path.join(result_v, "image")
result_test_num = os.path.join(result_test, "image")

image_list = os.listdir(image_A)
train_list = []
val_list = []
test_list = []

for image in random.sample((image_list), int(len(image_list) * 0.8)):
    train_list.append(image)

else_list = [i for i in image_list if i not in train_list]

for image in random.sample((else_list), int(len(else_list) * 0.5)):
    val_list.append(image)

test_list = [i for i in else_list if i not in val_list]

num = len(os.listdir(result_train_num))
print(num)
for i in train_list:
    num += 1
    shutil.copy(os.path.join(image_A, i), os.path.join(result_t, "image/train_{}.png".format(num)))

    shutil.copy(os.path.join(image_L, i.replace("tif","tif")), os.path.join(result_t, "label/train_{}.png".format(num)))

num = len(os.listdir(result_val_num))
for i in val_list:
    num += 1
    shutil.copy(os.path.join(image_A, i), os.path.join(result_v, "image/val_{}.png".format(num)))

    shutil.copy(os.path.join(image_L, i.replace("tif","tif")), os.path.join(result_v, "label/val_{}.png".format(num)))

num = len(os.listdir(result_test_num))
for i in test_list:
    num += 1
    shutil.copy(os.path.join(image_A, i), os.path.join(result_test, "image/test_{}.png".format(num)))

    shutil.copy(os.path.join(image_L, i.replace("tif","tif")), os.path.join(result_test, "label/test_{}.png".format(num)))
