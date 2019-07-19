# -*- coding: utf-8 -*-
# @Time    : 2019/6/20 15:28
# @Author  : ljf
import os
from PIL import Image
import glob
import time

from_path = "0701易分错图像"
# 读取图片名称为原来的label，先存
from_folder_list = os.listdir(from_path)
for folder in from_folder_list:
    from_image_paths = glob.glob(os.path.join(from_path,folder,"*.tif"))
    for img_path in from_image_paths:
        # 旧文件夹到新文件夹，名称为旧，现在文件夹为新
        new_folder = img_path.split("\\")[-2]
        # print(img_path)
        if img_path.split("\\")[-1].split(".")[0][:1] == "点"  or img_path.split("\\")[-1].split(".")[0][:2] == "粉尘":
            old_folder = "点粉"
        else:
            old_folder = img_path.split("\\")[-1].split(".")[0][:2]
        image_name = img_path.split("\\")[-1]
        # print(img_path.split("\\"))
        if old_folder == new_folder:
            print("前后标签一致，不用移动！")
            # print(img_path.split("\\"))
            continue
        try:
            # 移动到新文件夹，删除就文件夹中的图像
            new_img_path = os.path.join("./classification-7-619",new_folder,image_name)
            old_img_path = os.path.join("./classification-7-619",old_folder,image_name)
            image = Image.open(old_img_path)
            # print(image.size)
            image.save(new_img_path)
            time.sleep(0.5)
            print("将图片{}移动到{}".format(old_img_path,new_img_path))
            os.remove(old_img_path)

        except:
            print("{}图片读取出现问题".format(old_img_path))
        # print(new_folder)
        # print(old_folder)
    # print(from_image_paths)

# 现在的文件夹是新的label，再删除
