# -*- coding: utf-8 -*-
# @Time    : 2019/8/8 13:35
# @Author  : ljf
import torch
from torch import nn
from torch import optim
import numpy as np

# TODO
# 一 数据
train_x = torch.rand(size=[10,3,8,8])
train_y = torch.rand(size=[10,3,8,8])
# np.random.seed(18)
temp_x = [[1,2,3,4,5,6,7,8],
          [-1,-2,-3,-4,-5,-6,-7,-8],
          [1,2,3,4,5,6,7,8],
          [-1,-2,-3,-4,-5,-6,-7,-8],
          [1, 2, 3, 4, 5, 6, 7, 8],
          [-1, -2, -3, -4, -5, -6, -7, -8],
          [1, 2, 3, 4, 5, 6, 7, 8],
          [-1, -2, -3, -4, -5, -6, -7, -8]]
temp_y = np.array([[temp_x,temp_x,temp_x]])
test_x = torch.Tensor(temp_y)
print(test_x.size())
# print(test_x)
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.bn = nn.BatchNorm2d(num_features=3)
        # for m in self.modules():
        #     if isinstance(m,nn.Conv2d):
        #         nn.init.constant_(m.weight,1.0)
        #         nn.init.constant_(m.bias,1.0)
    def forward(self, x):
        out = self.bn(x)
        return out
# 三 优化器，损失函数
is_evaluate = True
model = Net()
if is_evaluate:
    model.load_state_dict(torch.load("./pth/batchnorm2d.pth"))
    mean = test_x.mean(dim=[2, 3], keepdim=True)
    var = test_x.var(dim=[2, 3], keepdim=True)
    # print(model.bn.running_mean)
    _out = (test_x - model.bn.running_mean.view(1, 3, 1, 1)) / torch.sqrt(
        model.bn.running_var.view(1, 3, 1, 1) + model.bn.eps)
    _output = model.bn.weight.view(mean.size()) * _out + model.bn.bias.view(mean.size())
    print(_output)
    model.eval()
    # print(model.bn.eps)
    pred_y = model(test_x)
    print(pred_y)
else:
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    # 四 迭代数据
    for i in range(20):

        output = model(train_x)
        if i ==0:
            print(output.size())
        loss = criterion(output, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 五 模型保存

    torch.save(model.state_dict(),"./pth/batchnorm2d.pth")

