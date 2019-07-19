# -*- coding: utf-8 -*-
# @Time    : 2019/6/19 11:35
# @Author  : ljf
import glob
import os
from PIL import Image
import time

data_list = ["classification-8-424","classification-8-527"]
folder_list = ["点","良品","丝印","毛丝","脏污","粉尘","划伤","断胶"]
# 1.遍历两个数据集中的文件夹
for folder in folder_list:
    num = 0
    folder_path = "data/classification-7-619/{}".format(folder)
    time.sleep(3)
    print("开始处理{}数据".format(folder))
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    for data in data_list:
        time.sleep(2)
        print("数据集：{}".format(data))
        imgs_path = glob.glob(os.path.join("data",data,folder,"*.tif"))
        for path in imgs_path:
            img = Image.open(path)
            img.save(os.path.join(folder_path,"{}.tif".format(folder+str(num))))
            num+=1
    print("{}类别数据共：{}张".format(folder,num))
# 2.创建文件夹并将数据重新命名放入文件夹中

