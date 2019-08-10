# -*- coding: utf-8 -*-
# @Time    : 2019/8/8 18:34
# @Author  : ljf
import torch
from torch import nn
from torch import optim
import numpy as np

# TODO
# 一 数据
train_x = torch.rand(size=[10,3,8,8])
train_y = torch.rand(size=[10,3,4,4])
np.random.seed(18)
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
        self.conv2d = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, bias=True, stride=2, padding=1)
        self.leakyrelu = nn.LeakyReLU()
        # for m in self.modules():
        #     if isinstance(m,nn.Conv2d):
        #         nn.init.constant_(m.weight,1.0)
        #         nn.init.constant_(m.bias,1.0)
    def forward(self, x):
        out = self.conv2d(x)
        out = self.leakyrelu(out)
        return out

# 三 优化器，损失函数
is_evaluate = True
model = Net()
if is_evaluate:
    model.load_state_dict(torch.load("./leakyrelu.pth"))
    # test_x = torch.Tensor(np.random.randint(1,9,(1,10,3,3)))
    # for k,v in torch.load("leakyrelu.pth").items():
    #    print(v)
    # print(test_x)
    model.eval()
    pred_y = model(test_x)
    print(pred_y)
else:
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    # for k,v in model.state_dict().items():
    #     print(type(k))
    #     print(v.size())
    print(sum([p.numel() for p in model.parameters()]))
    criterion = nn.MSELoss()
    # 四 迭代数据
    for i in range(20):
        # print("***Epoch:{}***".format(i))
        output = model(train_x)
        if i ==0:
            print(output.size())
        loss = criterion(output, train_y)
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        # print("Loss:{}".format(loss))
    # 五 模型保存
    # [0.535030,0.461349,0.541562,0.550206,0.534486,]
    # import io
    torch.save(model.state_dict(),"./leakyrelu.pth")

