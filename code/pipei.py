import hashlib
import os
import shutil

def get_file_hash(file_path):
    hasher = hashlib.sha256()
    with open(file_path,'rb') as file:
        buffer = file.read()
        hasher.update(buffer)
    return hasher.hexdigest()

def get_images_info(folder):
    """获取文件夹中的所有图片路径和大小"""
    images_info = {}
    for root,_,files in os.walk(folder):
        for file in files:
            file_path = os.path.join(root,file)
            if file.lower().endswith(('png','jpg','tif')):
                file_size= os.path.getsize(file_path)
                if file_size in images_info:
                    images_info[file_size].append(file_path)
                else:
                    images_info[file_size]=[file_path]
    return images_info

def find_matching_images(folder1_images,folder2_images):
    matching_images = []

    for file_hash,paths2 in folder2_images.items():
        if file_hash in folder1_images:
            paths1 = folder1_images[file_hash]
            for path1 in paths1:
                for path2 in paths2:
                    if os.path.basename(path1) != os.path.basename(path2):
                        matching_images.append((path1,path2))
    return matching_images

folder1 = r"F:\XJ\python_code\segmentation_baseline\data\dataset4\OUT1\image"
folder2 = r"F:\XJ\python_code\segmentation_baseline\data\dataset3\test\image"
destination_folder = r"F:\XJ\python_code\segmentation_baseline\data\dataset4\test"
log_file = r"F:\XJ\python_code\segmentation_baseline\data\dataset4\test.txt"

if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

folder1_images = get_images_info(folder1)
print(folder1_images)
folder2_images = get_images_info(folder2)

matching_images =find_matching_images(folder1_images,folder2_images)

with open(log_file,'w') as log:
    for image1,image2 in matching_images:
        new_name = os.path.basename(image2)
        new_path = os.path.join(destination_folder,new_name)
        shutil.move(image1,new_path)
        log.write(f'{os.path.basename(image1)} {new_name}\n')
