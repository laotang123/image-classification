# -*- coding: utf-8 -*-
# @Time    : 2019/7/30 14:02
# @Author  : ljf
import torch
from torch import nn
from torch import optim
import numpy as np

# TODO
# 一 数据
train_x = torch.rand(size=[5,5])
train_y = torch.rand(size=[5,1])
test_x = torch.Tensor(np.array([ -1., -2., -3., -4., -5.,
					1.,2.,3.,4.,5.,
					 -1., -2., -3., -4., -5.,
					1.,2.,3.,4.,5.,
					1.,2.,3.,4.,5.,]).reshape(5,5))
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
        self.fc1 = nn.Linear(5,10,bias=False)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(10,25)
        self.sigmoid = nn.Sigmoid()
        self.fc3 = nn.Linear(25,1)
        self.prob = nn.Sigmoid()
    def forward(self, x):
        out = self.fc1(x)
        out = self.tanh(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = self.fc3(out)
        out = self.prob(out)
        return out
# 三 优化器，损失函数
is_evaluate = True
model = Net()
if is_evaluate:
    model.load_state_dict(torch.load("./weight-linear.pth"))
    # test_x = torch.Tensor(np.random.randint(1,9,(1,5)))
    # for k,v in torch.load("./weight-conv2d-bn.pth").items():
    #     print(k)
    #     if k == "bn.num_batches_tracked":
    #         print(v)
    #         continue
    #     print(v.size())
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
        print(output.size())
        loss = criterion(output, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print("Loss:{}".format(loss))
    # 五 模型保存
    # [0.535030,0.461349,0.541562,0.550206,0.534486,]
    # import io
    torch.save(model.state_dict(),"./weight-linear.pth")
    # buffer = io.BytesIO()
    # torch.save(model.state_dict(), buffer)

