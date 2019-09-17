# -*- coding: utf-8 -*-
# @Time    : 2019/5/29 15:30
# @Author  : ljf
import glob
import os
from PIL import Image

def ProcessMagnetic():
    data_path = "data/magnetic"
    new_data_path = "data/magnetic-ljf"
    folder_paths = os.listdir(data_path)

    for folder in folder_paths:
        if not os.path.isdir(os.path.join(data_path, folder)):
            continue
        new_folder_path = os.path.join(new_data_path, folder)
        if not os.path.exists(new_folder_path):
            os.mkdir(new_folder_path)
        img_paths = glob.glob(os.path.join(data_path, folder, "*.jpg"))
        for path in img_paths:
            img = Image.open(path, mode="r")
            img_name = path.split("\\")[-1].split(".")[-2]
            resize_img = img.resize((480, 480))
            resize_img.save(new_folder_path + "\\" + img_name + ".jpg")


def ProcessDagn():
    """
    将dagn数据集处理为ok，gn。十个数据集的二分类以及一个11分类的数据集
    :return:
    """
    for i in range(1,11):
        for mode in ["Train","Test"]:
            data_path = "data/DAGN/Class{}/{}".format(i,mode)
            new_data_path = "data/dagn-ljf/Class{}/{}"

            # 检查创建新文件夹
            for folder in ["ok","ng"]:
                if not os.path.exists(new_data_path.format(i,folder)):
                    os.makedirs(new_data_path.format(i,folder))
            label_path = data_path + "/" + "Label"
            ng_img_paths = glob.glob(label_path + "/" + "*.PNG")
            ng_img_name_list = [ng_path.split("\\")[-1].split(".")[-2].split("_")[-2] for ng_path in ng_img_paths]
            for path in glob.glob(os.path.join(data_path,"*.PNG")):
                img_name = path.split("\\")[-1].split(".")[-2]

                img = Image.open(path, mode="r")
                # 在label文件下的图像为负样本
                if img_name in ng_img_name_list:
                    img.save(new_data_path.format(i,"ng")+"/"+img_name+".jpg")
                else:
                    img.save(new_data_path.format(i,"ok")+"/"+img_name+".jpg")
        print("已完成第{}个数据集".format(i))

def ProcessDagn11():
    """
    将数据集dagn的十个ng+一个ok一共十一个类别
    :return: 
    """
    data_path = "data/dagn-ljf"
    for i in range(1,11):
        for mode in ["ok","ng"]:
            id = 0
            img_paths = glob.glob(data_path+"/Class{}/{}/*.jpg".format(i,mode))
            if mode == "ng":
                folder_path = data_path+"/Class11/"+"c{}-ng".format(i)
            else:
                folder_path = data_path + "/Class11/" + "ok"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            for path in img_paths:
                img_name = path.split("\\")[-1].split(".")[-2]
                img = Image.open(path,mode="r")
                img.save(folder_path+"/"+img_name+".jpg")
                # 每个类别选30张ok图片
                if mode == "ok":
                    if id > 30:
                        break
                    id += 1

        print("已经处理完第{}个数据集".format(i))


if __name__ == "__main__":
    # ProcessDagn()
    ProcessDagn11()
