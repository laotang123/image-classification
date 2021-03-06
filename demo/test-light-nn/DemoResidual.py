# -*- coding: utf-8 -*-
# @Time    : 2019/8/15 14:00
# @Author  : ljf
import torch
from torch import nn
from torch import optim
import numpy as np

# TODO layer0的卷积层输出结果不对！
# 一 数据
train_x = torch.rand(size=[10, 3, 8, 8])
train_y = torch.rand(size=[10, 16, 8, 8])
np.random.seed(18)
temp_x = [[1, 2, 3, 4, 5, 6, 7, 8],
          [-1, -2, -3, -4, -5, -6, -7, -8],
          [1, 2, 3, 4, 5, 6, 7, 8],
          [-1, -2, -3, -4, -5, -6, -7, -8],
          [1, 2, 3, 4, 5, 6, 7, 8],
          [-1, -2, -3, -4, -5, -6, -7, -8],
          [1, 2, 3, 4, 5, 6, 7, 8],
          [-1, -2, -3, -4, -5, -6, -7, -8]]
temp_y = np.array([[temp_x, temp_x, temp_x]])
test_x = torch.Tensor(temp_y)
print(test_x.size())
# print(test_x)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):  # 基本模块
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, depth):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = BasicBlock

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.LeakyReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, n)
        # self.layer2 = self._make_layer(block, 32, n, stride=2)
        # for m in self.modules():
        #     if isinstance(m,nn.Conv2d):
        #         nn.init.constant_(m.weight,1.0)
        #         nn.init.constant_(m.bias,1.0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 32x32

        x = self.layer1(x)  # 32x32
        # x = self.layer2(x)
        return x


# 三 优化器，损失函数
is_evaluate = True
model = ResNet(20)
if is_evaluate:
    model.load_state_dict(torch.load("./pth/residual1.pth"))
    # test_x = torch.Tensor(np.random.randint(1,9,(1,10,3,3)))
    # for k, v in torch.load("./pth/residual.pth").items():
    #     print(k)
    # print(test_x)
    model.eval()
    print(model.conv1.weight.size())
    # print(model.conv1(test_x).size())
    # print("model.conv1" + "*" * 200)
    # temp = model.conv1(test_x)
    # print(temp)
    # print("model.bn1" + "*" * 200)
    # temp = model.bn1(temp)
    # print(temp)
    # print("model.relu" + "*" * 200)
    # temp = model.relu(temp)
    # print(temp)
    # print(model.layer)
    # print("model.layer.conv1" + "*" * 200)
    # temp = model.layer[0].conv1(temp)
    # print(temp)
    # print("model.layer.bn1" + "*" * 200)
    # temp = model.layer[0].bn1(temp)
    # print(temp)
    # print("model.layer.relu1" + "*" * 200)
    # temp = model.layer[0].relu(temp)
    # print(temp)
    # pred_y = model(test_x)
    # print(pred_y.size())
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
        if i == 0:
            print(output.size())
        loss = criterion(output, train_y)
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        # print("Loss:{}".format(loss))
    # 五 模型保存
    # [0.535030,0.461349,0.541562,0.550206,0.534486,]
    # import io
    # torch.save(model.state_dict(), "./pth/residual1.pth")
