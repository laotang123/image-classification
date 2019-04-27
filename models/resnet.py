# encoding=utf8
from __future__ import absolute_import

import torch.nn as nn
import math
from models.OctConv_main import *


def conv3x3(
        in_planes,
        out_planes,
        stride=1,
        alpha_out=0.5,
        is_octconv=False,
        octconv_type=""):
    "3x3 convolution with padding"
    return Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=(
            1,
            1),
        bias=False,
        alpha_out=alpha_out,
        is_octconv=is_octconv,
        octconv_type=octconv_type)


class BasicBlock(nn.Module):  # 基本模块
    expansion = 1

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            downsample=None,
            alpha_out=0.5,
            last_conv=False,
            is_octconv=False,
            octconv_type=""):
        super(BasicBlock, self).__init__()
        l_planes = int(planes * alpha_out)
        h_planes = planes - l_planes
        # self.last_conv = last_conv
        self.conv1 = conv3x3(
            inplanes,
            planes,
            stride,
            is_octconv=is_octconv,
            octconv_type=octconv_type)
        self.bn1 = BatchNorm2d(l_planes, h_planes, is_octconv)
        self.relu1 = AC(name="relu", inplace=True, is_octconv=is_octconv)
        self.relu2 = AC(name="relu", inplace=True, is_octconv=is_octconv)
        if not is_octconv:
            self.conv2 = conv3x3(planes, planes)
            self.bn2 = BatchNorm2d(l_planes, h_planes)
        else:
            if last_conv:
                self.conv2 = conv3x3(
                    planes,
                    planes,
                    alpha_out=0,
                    is_octconv=is_octconv,
                    octconv_type=octconv_type)
                self.bn2 = nn.BatchNorm2d(planes)
                self.relu2 = nn.ReLU(inplace=True)
            else:
                self.conv2 = conv3x3(
                    planes,
                    planes,
                    is_octconv=is_octconv,
                    octconv_type=octconv_type)
                self.bn2 = BatchNorm2d(l_planes, h_planes, is_octconv)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if isinstance(out, tuple):
            out[0].add_(residual[0])
            out[1].add(residual[1])
        else:
            out += residual
        out = self.relu2(out)
        nn.LeakyReLU()
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            downsample=None,
            alpha_out=0.5,
            last_conv=False,
            is_octconv=False,
            octconv_type=""):
        super(Bottleneck, self).__init__()
        l_planes = int(planes * alpha_out)
        h_planes = planes - l_planes
        self.last_conv = last_conv
        self.conv1 = Conv2d(
            inplanes,
            planes,
            kernel_size=1,
            bias=False,
            is_octconv=is_octconv,
            octconv_type=octconv_type)
        self.bn1 = BatchNorm2d(l_planes, h_planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=stride,
                            padding=1, bias=False, is_octconv=is_octconv)
        self.bn2 = BatchNorm2d(l_planes, h_planes)
        self.relu = AC(name="relu", inplace=True, is_octconv=is_octconv)
        if not is_octconv:
            self.conv3 = conv3x3(planes, planes)
            self.bn3 = BatchNorm2d(l_planes, h_planes)
        else:
            if self.last_conv:
                self.conv3 = conv3x3(
                    planes,
                    planes,
                    alpha_out=0,
                    is_octconv=is_octconv,
                    octconv_type=octconv_type)
                self.bn3 = nn.BatchNorm2d(planes)
                self.relu3 = nn.ReLU(inplace=True)
            else:
                self.conv3 = conv3x3(
                    planes,
                    planes,
                    is_octconv=is_octconv,
                    octconv_type=octconv_type)
                self.bn3 = BatchNorm2d(l_planes, h_planes, is_octconv)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        #out = self.relu(x)
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
        if isinstance(out, tuple):
            out[0].add_(residual[0])
            out[1].add(residual[1])
        else:
            out += residual
        out = self.relu3(out)

        return out


class ResNet(nn.Module):

    def __init__(self, depth, args, num_classes=1000):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = Bottleneck if depth >= 44 else BasicBlock

        self.inplanes = 16
        self.alpha_out = args.alpha_out
        self.conv1 = Conv2d(
            3,
            16,
            kernel_size=3,
            padding=(
                1,
                1),
            bias=False,
            alpha_in=0,
            alpha_out=self.alpha_out,
            is_octconv=args.is_octconv,
            octconv_type=args.octconv_type)
        self.bn1 = BatchNorm2d(int(16 * self.alpha_out),
                               16 - int(16 * self.alpha_out), args.is_octconv)
        self.relu = AC("leakyrelu", True, args.is_octconv)

        self.layer1 = self._make_layer(
            block,
            16,
            n,
            is_octconv=args.is_octconv,
            octconv_type=args.octconv_type)
        self.layer2 = self._make_layer(
            block,
            32,
            n,
            stride=2,
            is_octconv=args.is_octconv,
            octconv_type=args.octconv_type)
        self.layer3 = self._make_layer(
            block,
            64,
            n,
            stride=2,
            is_octconv=args.is_octconv,
            octconv_type=args.octconv_type,
            last_conv=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        # print(list(self.modules()))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(
            self,
            block,
            planes,
            blocks,
            stride=1,
            last_conv=False,
            is_octconv=False,
            octconv_type=""):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            l_planes = int(planes * block.expansion * self.alpha_out)
            h_planes = planes * block.expansion - l_planes
            # if last_conv:
            # downsample = nn.Sequential(
            # Conv2d(self.inplanes, planes * block.expansion,
            # kernel_size=1, stride=stride, bias=False, alpha_out=0,is_octconv=is_octconv, octconv_type=octconv_type),
            # nn.BatchNorm2d(l_planes+h_planes))
            # else:
            downsample = nn.Sequential(
                Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                    is_octconv=is_octconv,
                    octconv_type=octconv_type),
                BatchNorm2d(
                    l_planes,
                    h_planes,
                    is_octconv))
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                alpha_out=self.alpha_out,
                is_octconv=is_octconv,
                octconv_type=octconv_type))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == blocks - 1 and last_conv:
                octconv_residual = nn.Sequential(
                    Conv2d(
                        self.inplanes,
                        planes *
                        block.expansion,
                        kernel_size=1,
                        bias=False,
                        alpha_out=0,
                        is_octconv=is_octconv,
                        octconv_type=octconv_type),
                    nn.BatchNorm2d(
                        planes *
                        block.expansion))
                layers.append(
                    block(
                        self.inplanes,
                        planes,
                        alpha_out=self.alpha_out,
                        downsample=octconv_residual,
                        last_conv=last_conv,
                        is_octconv=is_octconv,
                        octconv_type=octconv_type))
            else:
                layers.append(
                    block(
                        self.inplanes,
                        planes,
                        alpha_out=self.alpha_out,
                        is_octconv=is_octconv,
                        octconv_type=octconv_type))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)
