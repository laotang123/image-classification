# -*- coding: utf-8 -*-
# @Time    : 2019/8/21 16:40
# @Author  : ljf
import numpy as np
from PIL import Image
from torchvision import transforms
from array import *
import glob

def format_name(name):
    if len(name) < 16:
        return name.ljust(16, '\0')
def gen_data(img_paths):
    transform_test = transforms.Compose([
        # transforms.RandomCrop(480, padding=10),
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        # transforms.Normalize((0.5624, 0.5624, 0.5624),
        #                      (0.4172, 0.4172, 0.4172)),
        # transforms.Normalize((0.5464, 0.5464, 0.5464),
        #                      (0.4148, 0.4148, 0.4148))
        transforms.Normalize((0.5544, 0.5544, 0.5544),
                             (0.4160, 0.4160, 0.4160))
    ])
    val_arr = array('f')
    # size_arr = array('I')
    file = open("images.dat", "wb")
    batch_size = 2000
    for id, img_path in enumerate(glob.glob(img_paths)):
        image = Image.open(img_path).convert("RGB")
        img_list = transform_test(image).numpy().flatten().tolist()
        if id == batch_size:
            break
        # print(img_path)
        val_arr.fromlist(img_list)
    # size_arr.append(batch_size)
    # size_arr.append(len(img_list))
    # size_arr.tofile(file)
    val_arr.tofile(file)
    file.close()
if __name__ == "__main__":
    import time
    start = time.time()
    gen_data("D:\\git-projects\\ljf-git\\image-classification\\demo\\data\\enhance-classification-7-730\\ddust\\*.tif")
    print("平均存储一张照片时间：{}s".format((time.time()-start)/128))