# -*- coding: utf-8 -*-
# @Time    : 2019/7/17 9:40
# @Author  : ljf
import torch
from torch import nn
from torch import optim

# 一 数据
train_x = torch.rand(size=[10,3,5,5])
train_y = torch.rand(size=[10,5,1,1])
# 二 网络
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=10,kernel_size=3,stride=1)
        self.tanh = nn.Tanh()
        self.conv2 = nn.Conv2d(in_channels=10,out_channels=5,kernel_size=3,stride=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.conv1(x)
        out = self.tanh(out)
        out = self.conv2(out)
        out = self.sigmoid(out)
        return out
# 三 优化器，损失函数
model = Net()
optimizer = optim.SGD(model.parameters(),lr=0.001)
# for k,v in model.state_dict().items():
#     print(type(k))
#     print(v.size())
print(sum([p.numel() for p in model.parameters()]))
criterion = nn.MSELoss()
# 四 迭代数据
for i in range(20):
    # print("***Epoch:{}***".format(i))

    pred_y = model(train_x)
    print(pred_y.size())
    loss = criterion(pred_y,train_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # print("Loss:{}".format(loss))
# 五 模型保存
# import io
torch.save(model.state_dict(),"./weight-conv2d.pth")
# buffer = io.BytesIO()
# torch.save(model.state_dict(), buffer)
