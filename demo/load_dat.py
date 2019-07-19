# -*- coding: utf-8 -*-
# @Time    : 2019/5/29 15:30
# @Author  : ljf

import json
import torch
import numpy as np

# checkpoint = torch.load("./weight.pth")
# print(checkpoint)
print(np.fromfile("./data/weight-ljf.dat",dtype=float).shape)
print(np.fromfile("./data/weight.dat",dtype=float).shape)

