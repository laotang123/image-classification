# -*- coding: utf-8 -*-
# @Time    : 2019/8/27 11:39
# @Author  : ljf
from __future__ import absolute_import

import torch.nn as nn
import math
import torch
import numpy as np
from torch import optim
from collections import OrderedDict
from PIL import Image
from torchvision import transforms

np.set_printoptions(suppress=True,threshold=np.inf)
# 一 数据
train_x = torch.rand(size=[10, 3, 8, 8])
train_y = torch.randint(low=0,high=7,size=[10])
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
id = 1
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

    def __init__(self, depth, num_classes=1000):
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

        self.layer1 = self._make_layer(block, 16, n, stride=2)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        # self.layer3 = self._make_layer(block, 64, n, stride=2)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.bn2 = nn.BatchNorm2d(32 * block.expansion)
        self.fc = nn.Linear(32 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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
        x = self.layer2(x)  # 16x16
        # x = self.layer3(x)  # 8x8
        # x = self.avgpool(x)
        x = self.maxpool(x)
        # print(x.detach().numpy())
        # print(x.size())

        # t_x = x[0][0][0][0]
        # print("pooling后的特征图：\n{}".format(x.view(-1)))
        # print(self.bn2.weight[0] * ((t_x-self.bn2.running_mean[0])/torch.sqrt(self.bn2.running_var[0]+1e-5))+self.bn2.bias[0])
        # print(self.bn2.weight[1] * ((x[0][1][0][0] - self.bn2.running_mean[1]) / torch.sqrt(self.bn2.running_var[1] + 1e-5)) +
        #       self.bn2.bias[1])
        x = self.bn2(x)

        # print(x[0][0][0][0])
        # print(x[0][1][0][0])
        # print("bn后的特征图：\n{}".format(x.view(-1).detach().numpy()))
        # print("bn的weight：{}".format(self.bn2.weight ))
        # print("bn的bias：{}".format(self.bn2.bias))
        # print("bn的running_mean：{}".format(self.bn2.running_mean))
        # print("bn的running_var：{}".format(self.bn2.running_var))
        #
        #
        # print("输出Linear层的权重：")
        # print("linear的wight：{}".format(self.fc.weight))
        # print("linear的bias：{}".format(self.fc.bias))
        # print(x.detach().numpy())
        # print(x.size())
        # print(self.bn2.weight[0])
        # print(self.bn2.bias[0])
        # print(self.bn2.running_mean[0])
        # import time
        # time.sleep(1000)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


is_evaluate = True
model = ResNet(20, 7)


if is_evaluate:
    checkpoint = torch.load("./pth/sgd-depth14-ao_resnet-lr0.5-2019-08-27-09_37_09.978241checkpoint.pth.tar",map_location="cpu")["state_dict"]
    # print(checkpoint["state_dict"])
    state_dict = OrderedDict()
    # print(checkpoint["state_dict"]["module.conv1.weight"])
    for k in checkpoint.keys():
        new_k = k.replace("module.","")
        state_dict[new_k] = checkpoint[k]
    model.load_state_dict(state_dict)
    # model.load_state_dict(torch.load("./pth/demo_resnet20.pth"))

    transform_test = transforms.Compose([
        # transforms.RandomCrop(480, padding=10),
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        # transforms.Normalize((0.5624, 0.5624, 0.5624),
        #                      (0.4172, 0.4172, 0.4172)),
        # transforms.Normalize((0.5464, 0.5464, 0.5464),
        #                      (0.4148, 0.4148, 0.4148))
        transforms.Normalize((0.5544, 0.5544, 0.5544),
                             (0.4160, 0.4160, 0.4160))
    ])
    img_path = "D:\\git-projects\\ljf-git\\image-classification\\demo\\data\\enhance-classification-7-730\\ddust\\ddust1-.tif"
    image = Image.open(img_path).convert("RGB")
    # print(transform_test(image).numpy()[0][0:100])
    img = torch.unsqueeze(transform_test(image), 0)
    model.eval()
    model(img)
    print(model(img))
    # print(model.conv1(test_x).size())
    # print("model.conv1" + "*" * 200)
    # temp = model.conv1(test_x)
    # # print(temp)
    # print("model.bn1" + "*" * 200)
    # temp = model.bn1(temp)
    # # print(temp)
    # print("model.relu" + "*" * 200)
    # relu1 = model.relu(temp)
    # # print(relu1)
    # print("model.layer1.conv1" + "*" * 200)
    # temp = model.layer1[0].conv1(relu1)
    # # print(temp)
    # print("model.layer1.bn1" + "*" * 200)
    # temp = model.layer1[0].bn1(temp)
    # # print(temp)
    # print("model.layer1.relu1" + "*" * 200)
    # temp = model.layer1[0].relu(temp)
    # # print(temp)
    # print("model.layer1.conv2" + "*" * 200)
    # temp = model.layer1[0].conv2(temp)
    # # print(temp)
    # print("model.layer1.bn2" + "*" * 200)
    # temp = model.layer1[0].bn2(temp)
    # # print(temp)
    # temp += relu1
    # print("model.layer1.relu2" + "*" * 200)
    # temp = model.layer1[0].relu(temp)
    # print(temp)
    # temp = model.conv1(test_x)
    # temp = model.bn1(temp)
    # temp = model.relu(temp)
    # # temp = model.layer1(temp)
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
    criterion = nn.CrossEntropyLoss()
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
    # torch.save(model.state_dict(), "./pth/demo_resnet20.pth")