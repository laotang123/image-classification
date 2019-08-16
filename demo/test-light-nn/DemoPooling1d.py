# -*- coding: utf-8 -*-
# @Time    : 2019/8/9 13:25
# @Author  : ljf
import torch
from torch import nn
from torch import optim
import numpy as np

# TODO
# 一 数据
np.random.seed(18)
temp_x = [[1,2,3,4,5,6,7,8],
          [-1,-2,-3,-4,-5,-6,-7,-8],
          [1,2,3,4,5,6,7,8],
          [-1,-2,-3,-4,-5,-6,-7,-8],
          [1, 2, 3, 4, 5, 6, 7, 8],
          [-1, -2, -3, -4, -5, -6, -7, -8],
          [1, 2, 3, 4, 5, 6, 7, 8],
          [-1, -2, -3, -4, -5, -6, -7, -8]]
temp_y = np.array(temp_x)
test_x = torch.Tensor([temp_y])
print(test_x.size())
# print(test_x)
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.pooling1d = nn.MaxPool1d(kernel_size=3,stride=2,padding=1)
        # for m in self.modules():
        #     if isinstance(m,nn.Conv2d):
        #         nn.init.constant_(m.weight,1.0)
        #         nn.init.constant_(m.bias,1.0)
    def forward(self, x):
        out = self.pooling1d(x)
        return out

# 三 优化器，损失函数
is_evaluate = True
model = Net()
print(test_x)
out = model(test_x)
print(out)
print(out.size())
