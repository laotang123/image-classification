# -*- coding: utf-8 -*-
# @Time    : 2019/8/8 10:45
# @Author  : ljf
import torch
from array import *
import numpy as np

np.set_printoptions(suppress=True)
torch.manual_seed(18)
var_size = array("f")
tensor = torch.Tensor([1,2,3,4,5,6,7,8,9]).flatten().numpy().astype(np.float16).tolist()
var_size.fromlist(tensor)
file = open("image-fp16.dat","wb")
var_size.tofile(file)
# weights = torch.load("./test-light-nn/pth/sgd-depth14-ao_resnet-lr0.5-2019-08-27-09_37_09.978241model_best.pth.tar")
# print(weights["state_dict"]["module.conv1.weight"].cpu().numpy().astype(np.float16))