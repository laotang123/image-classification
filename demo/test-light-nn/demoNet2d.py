# -*- coding: utf-8 -*-
# @Time    : 2019/7/17 9:40
# @Author  : ljf
import torch
from torch import nn
from torch import optim
import numpy as np

# TODO
# 一 数据
train_x = torch.rand(size=[10,3,5,5])
train_y = torch.rand(size=[10,5,1,1])
# test_x = torch.Tensor(np.array([ -1., -2., -3., -4., -5.,
# 					1.,2.,3.,4.,5.,
# 					 -1., -2., -3., -4., -5.,
# 					1.,2.,3.,4.,5.,
# 					1.,2.,3.,4.,5.,
# 					-1., -2., -3., -4., -5.,
# 					1.,2.,3.,4.,5.,
# 					 -1., -2., -3., -4., -5.,
# 					1.,2.,3.,4.,5.,
# 					1.,2.,3.,4.,5.,
# 					-1., -2., -3., -4., -5.,
# 					1.,2.,3.,4.,5.,
# 					 -1., -2., -3., -4., -5.,
# 					1.,2.,3.,4.,5.,
# 					1.,2.,3.,4.,5.]).reshape([1,3,5,5]))
# print(test_x)
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
is_evaluate = False
model = Net()
if is_evaluate:
    model.load_state_dict(torch.load("./weight-conv2d.pth"))
    test_x = torch.Tensor(np.random.randint(1,9,(1,10,3,3)))
    # for k,v in torch.load("./weight-conv2d-bn.pth").items():
    #     print(k)
    #     if k == "bn.num_batches_tracked":
    #         print(v)
    #         continue
    #     print(v.size())
    model.eval()
    print(model.bn(test_x))
    mean = test_x.mean(dim=[2,3],keepdim=True)
    var = test_x.var(dim=[2,3],keepdim=True)
    print(model.bn.running_mean)
    _out = (test_x-model.bn.running_mean.view(1,10,1,1))/torch.sqrt(model.bn.running_var.view(1,10,1,1)+model.bn.eps)
    _output = model.bn.weight.view(mean.size())*_out + model.bn.bias.view(mean.size())
    print(_output)
    # pred_y = model(test_x)
    # print(pred_y)
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
        print(output.size())
        loss = criterion(output, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print("Loss:{}".format(loss))
    # 五 模型保存
    # [0.535030,0.461349,0.541562,0.550206,0.534486,]
    # import io
    torch.save(model.state_dict(),"./weight-conv2d.pth")
    # buffer = io.BytesIO()
    # torch.save(model.state_dict(), buffer)
