# -*- coding: utf-8 -*-
# @Time    : 2019/8/29 18:09
# @Author  : ljf
from PIL import Image

img = Image.open("D:\git-projects\ljf-git\image-classification\demo\data\pill\ginseng\crack\pill_ginseng_crack_001.png")
# img.show()
resize_img = img.resize((480,480))
resize_img.show()