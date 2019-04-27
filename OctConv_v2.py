# -*- coding: utf-8 -*-
# @Time    : 2019/4/22 10:35
# @Author  : ljf
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class OctConv2d_v2(nn.Conv2d):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            alpha_in=0.5,
            alpha_out=0.5,):
        assert alpha_in >= 0 and alpha_in <= 1
        assert alpha_out >= 0 and alpha_out <= 1
        super(OctConv2d_v2, self).__init__(in_channels, out_channels,
                                           kernel_size, stride, padding,
                                           dilation, groups, bias)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out
        self.inChannelSplitIndex = math.floor(
            self.alpha_in * self.in_channels)
        self.outChannelSplitIndex = math.floor(
            self.alpha_out * self.out_channels)
        if bias:
            self.hh_bias = self.bias[self.outChannelSplitIndex:]
            self.hl_bias = self.bias[:self.outChannelSplitIndex]
            self.ll_bias = self.bias[ :self.outChannelSplitIndex]
            self.lh_bias = self.bias[ self.outChannelSplitIndex:]
        else:
            self.hh_bias = None
            self.hl_bias = None
            self.ll_bias = None
            self.lh_bias = None
    def forward(self, input):
        if not isinstance(input, tuple):
            assert self.alpha_in == 0 or self.alpha_in == 1
            inputLow = input if self.alpha_in == 1 else None
            inputHigh = input if self.alpha_in == 0 else None
        else:
            inputLow = input[0]
            inputHigh = input[1]

        output = [0, 0]
        # H->H
        if self.outChannelSplitIndex != self.out_channels and self.inChannelSplitIndex != self.in_channels:
            outputH2H = F.conv2d(
                inputHigh,
                self.weight[
                    self.outChannelSplitIndex:,
                    self.inChannelSplitIndex:,
                    :,
                    :],
                self.hh_bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups)
            output[1] += outputH2H

        # H->L
        if self.outChannelSplitIndex != 0 and self.inChannelSplitIndex != self.in_channels:
            outputH2L = F.conv2d(
                self.avgpool(inputHigh),
                self.weight[
                    :self.outChannelSplitIndex,
                    self.inChannelSplitIndex:,
                    :,
                    :],
                self.hl_bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups)
            output[0] += outputH2L

        # L->L
        if self.outChannelSplitIndex != 0 and self.inChannelSplitIndex != 0:
            outputL2L = F.conv2d(
                inputLow,
                self.weight[
                    :self.outChannelSplitIndex,
                    :self.inChannelSplitIndex,
                    :,
                    :],
                self.ll_bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups)
            output[0] += outputL2L

        # L->H
        if self.outChannelSplitIndex != self.out_channels and self.inChannelSplitIndex != 0:
            outputL2H = F.conv2d(
                F.interpolate(inputLow, scale_factor=2),
                self.weight[
                    self.outChannelSplitIndex:,
                    :self.inChannelSplitIndex,
                    :,
                    :],
                self.lh_bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups)
            output[1] += outputL2H
        if isinstance(output[0],int):
            out = output[1]
        else:
            out = tuple(output)
        return out


if __name__ == "__main__":
    input = torch.randn(1, 3, 32, 32)
    octconv1 = OctConv2d(in_channels=3,
                         out_channels=6,
                         kernel_size=3,
                         stride=2,
                         padding=1,
                         dilation=1,
                         groups=1,
                         bias=True,
                         alpha_in=0.,
                         alpha_out=0.25)
    octconv2 = OctConv2d(in_channels=6,
                         out_channels=16,
                         kernel_size=3,
                         stride=1,
                         padding=1,
                         dilation=1,
                         groups=1,
                         bias=True,
                         alpha_in=0.25,
                         alpha_out=0.5)
    out = octconv1(input)
    print(len(out))
    print(out[0].shape)
    print(out[1].size())

    out = octconv2(out)
    print(len(out))
    print(out[0].size())
    print(out[1].size())
