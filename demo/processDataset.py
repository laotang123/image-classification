# -*- coding: utf-8 -*-
# @Time    : 2019/5/29 15:30
# @Author  : ljf
import glob
import os
from PIL import Image, ImageDraw,ImageFont
import json
from collections import defaultdict


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
    for i in range(1, 11):
        for mode in ["Train", "Test"]:
            data_path = "data/DAGN/Class{}/{}".format(i, mode)
            new_data_path = "data/dagn-ljf/Class{}/{}"

            # 检查创建新文件夹
            for folder in ["ok", "ng"]:
                if not os.path.exists(new_data_path.format(i, folder)):
                    os.makedirs(new_data_path.format(i, folder))
            label_path = data_path + "/" + "Label"
            ng_img_paths = glob.glob(label_path + "/" + "*.PNG")
            ng_img_name_list = [ng_path.split(
                "\\")[-1].split(".")[-2].split("_")[-2] for ng_path in ng_img_paths]
            for path in glob.glob(os.path.join(data_path, "*.PNG")):
                img_name = path.split("\\")[-1].split(".")[-2]

                img = Image.open(path, mode="r")
                # 在label文件下的图像为负样本
                if img_name in ng_img_name_list:
                    img.save(
                        new_data_path.format(
                            i, "ng") + "/" + img_name + ".jpg")
                else:
                    img.save(
                        new_data_path.format(
                            i, "ok") + "/" + img_name + ".jpg")
        print("已完成第{}个数据集".format(i))


def ProcessDagn11():
    """
    将数据集dagn的十个ng+一个ok一共十一个类别
    :return:
    """
    data_path = "data/dagn-ljf"
    for i in range(1, 11):
        for mode in ["ok", "ng"]:
            id = 0
            img_paths = glob.glob(data_path +
                                  "/Class{}/{}/*.jpg".format(i, mode))
            if mode == "ng":
                folder_path = data_path + "/Class11/" + "c{}-ng".format(i)
            else:
                folder_path = data_path + "/Class11/" + "ok"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            for path in img_paths:
                img_name = "class{}".format(
                    i) + path.split("\\")[-1].split(".")[-2]
                img = Image.open(path, mode="r")
                img.save(folder_path + "/" + img_name + ".jpg")
                # 每个类别选30张ok图片
                if mode == "ok":
                    if id > 30:
                        break
                    id += 1

        print("已经处理完第{}个数据集".format(i))


def ProcessGuangDongTrain():
    json_list = json.load(
        open(
            "data/guangdong1_round1_train2_20190828/Annotations/anno_train.json",
            "r"))
    img_path = "data/guangdong1_round1_train2_20190828/defect_Images"
    img2classes = defaultdict(set)
    # for i in range(len(json_list)):
    #     json_item = json_list[i]
    #
    #     print(json_item)
    #     img = Image.open(os.path.join(img_path, json_item['name']), mode="r")
    #     draw = ImageDraw.Draw(img)
    #     rectangle_xy = [
    #         (json_item['bbox'][0],
    #          json_item['bbox'][1]),
    #         (json_item['bbox'][2],
    #          json_item['bbox'][1]),
    #         (json_item['bbox'][2],
    #          json_item['bbox'][3]),
    #         (json_item['bbox'][0],
    #          json_item['bbox'][3])]
    #     # draw.polygon(rectangle_xy, outline=(255, 0, 0))
    #     font = ImageFont.truetype('simsun.ttc', 50)
    #     draw.text((60, 60), json_item['defect_name'], fill=(255, 0, 0), font=font)
    #     draw.rectangle(json_item['bbox'], outline=(255, 0, 0), width=3)
    #     img.show()
    # img.save("data/guangdong1_round1_train2_20190828/draw-boxes/{}".format(json_item['name']))
    for item in json_list:
        img2classes[item['name']].add(item['defect_name'])

    # 创建文件夹
    classes = set()
    for value in img2classes.values():
        for v in value:
            classes.add(v)
    # for c in classes:
    #     os.mkdir("data/cloth/{}".format(c))

    print(classes)
    # 过滤多标签图片，进行存储
    for item in img2classes.items():
        if len(item[1]) == 1:
            img = Image.open(os.path.join(img_path,item[0]))
            img.save("data/cloth/{}/{}".format(item[1].pop(),item[0]))

if __name__ == "__main__":
    # ProcessDagn()
    # ProcessDagn11()
    ProcessGuangDongTrain()
