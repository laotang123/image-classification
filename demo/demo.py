# -*- coding: utf-8 -*-
# @Time    : 2019/7/17 16:50
# @Author  : ljf
from torch import nn
import torch
import numpy as np

np.random.seed(18)
torch.manual_seed(18)
m = nn.BatchNorm2d(3,momentum=1)  # bn设置的参数实际上是channel的参数
input = torch.Tensor(np.random.randint(1,9,(1, 3, 2, 2)).astype(np.float32))
# input = torch.rand((1, 3, 2, 2))
output = m(input)
# m.momentum=1
 # print(output)
# a = (input[0, 0, :, :]+input[1, 0, :, :]+input[2, 0, :, :]+input[3, 0, :, :]).sum()/16
# b = (input[0, 1, :, :]+input[1, 1, :, :]+input[2, 1, :, :]+input[3, 1, :, :]).sum()/16
# c = (input[0, 2, :, :]+input[1, 2, :, :]+input[2, 2, :, :]+input[3, 2, :, :]).sum()/16
a = input[0,0,:,:].sum()/4
b = input[0,1,:,:].mean()
c = input[0,2,:,:].mean()
mean = input.mean(dim=[2,3],keepdim=True)
var = input.var(dim=[2,3],keepdim=True)
print('The mean value of the first channel is %f' % a.data)
print('The mean value of the first channel is %f' % b.data)
print('The mean value of the first channel is %f' % c.data)
print('The output mean value of the BN layer is %f, %f, %f' % (m.running_mean.data[0],m.running_mean.data[1],m.running_mean.data[2]))
# print(  m.running_mean)
# print(mean)
# print(m.running_var)
# print(var)
# print(input.size())
# print(input-m.running_mean.view(1,3,1,1))
# print(torch.sqrt(m.running_var+m.eps).view(1,3,1,1))
_out = (input-mean)/torch.sqrt(var+m.eps)
print(mean.size())
_output = m.weight.view(1,3,1,1)*_out+m.bias.view(1,3,1,1)
print(_output)
print(output)