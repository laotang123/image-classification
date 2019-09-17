# -*- coding: utf-8 -*-
# @Time    : 2019/9/3 8:33
# @Author  : ljf
import glob
from PIL import Image
import os
import math
import torch
from torch import nn

class ProcessImage(object):
    def __init__(self,root):
        self.root = root
        self.folder_list =  os.listdir(self.root)

    def Read(self,img_path):
        self.img =  Image.open(img_path)
    def Write(self,save_path):
        self.img.save(save_path)
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):  # 基本模块
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes // 4, stride)
        self.bn1 = nn.BatchNorm2d(planes // 4)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes // 4, planes // 4, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes // 4)

        self.conv3 = conv3x3(planes // 4, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
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
        n = 2

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
        # print(x)
        x = self.bn1(x)
        x = self.relu(x)  # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        # x = self.layer3(x)  # 8x8
        # x = self.avgpool(x)
        x = self.maxpool(x)
        x = self.bn2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



if __name__ == "__main__":
    # nums_per_folder = 100
    # process = ProcessImage("data/classification-7-730")
    # new_root = "data/meta-classification-7-100"
    # for folder in process.folder_list:
    #     new_folder = os.path.join(new_root,folder)
    #     id = 0
    #     if not os.path.exists(new_folder):
    #         os.mkdir(new_folder)
    #     for img_path in glob.glob(os.path.join(process.root,folder,"*.tif")):
    #         process.Read(img_path)
    #         # print(process.img)
    #         process.Write(new_folder+"/"+"{}{}.tif".format(folder,id))
    #         id += 1
    #         if id == nums_per_folder:
    #             break
    model = ResNet(20,7)
    state_dict = torch.load("results/meta-3.pth")
    print(type(state_dict))
    # print(state_dict['fc.weight'].requires_grad)
    weight = nn.Parameter(torch.ones(7,32))
    nn.init.kaiming_normal_(weight)
    bias = nn.Parameter(torch.zeros(7))
    state_dict['fc.weight'] = weight
    state_dict['fc.bias'] = bias

    print(state_dict['fc.weight'].requires_grad)
    # print(state_dict['fc.weight'].size())
    model.load_state_dict(state_dict)
    print(state_dict['bn1.weight'].requires_grad)