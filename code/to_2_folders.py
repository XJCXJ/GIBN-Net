import os
import shutil

JPG_folder = r"D:\Data\AnShu_result"
Paste_img_folder = r"D:\Data\AnShu_dataset\image"
Paste_label_folder = r"D:\Data\AnShu_dataset\label"
#  获取文件夹内的文件名
FileNameList = os.listdir(JPG_folder)
NewFileName = 0
for i in range(len(FileNameList)):
    #  判断当前文件是否为json文件
    if(os.path.splitext(FileNameList[i])[1] != ".json"):

         # 复制image文件
        jpg_file_name = FileNameList[i].split(".", 1)[0]
        img_file = JPG_folder + "\\" + jpg_file_name + "\\img.png"
        new_img_file = Paste_img_folder + "\\" + str(NewFileName) + ".png"
        shutil.copyfile(img_file, new_img_file)

        #  复制label文件
        jpg_file_name = FileNameList[i].split(".", 1)[0]
        label_file = JPG_folder + "\\" + jpg_file_name + "\\label.png"
        new_label_file = Paste_label_folder + "\\" + str(NewFileName) + ".png"
        shutil.copyfile(label_file, new_label_file)

        #  文件序列名+1
        NewFileName = NewFileName + 1