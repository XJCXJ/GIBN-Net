import os
import shutil

image_folder = r"F:\XJ\python_code\segmentation_baseline\data\dataset4\OUT\label"
destinaton_folder = r'F:\XJ\python_code\segmentation_baseline\data\dataset4\train\label'
txt_file = r"F:\XJ\python_code\segmentation_baseline\data\dataset4\train.txt"

if not os.path.exists(destinaton_folder):
    os.makedirs(destinaton_folder)

with open(txt_file,'r') as file:
    lines = file.readlines()

for line in lines:
    old_name,new_name = line.strip().split()
    old_path = os.path.join(image_folder,old_name)
    new_path = os.path.join(destinaton_folder,new_name)

    shutil.move(old_path,new_path)