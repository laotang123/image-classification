# -*- coding: utf-8 -*-
# @Time    : 2019/4/22 13:29
# @Author  : ljf
import torch
import torch.nn.functional as F
from torch import nn


class OctConv2d_v1(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 alpha_in=0.5,
                 alpha_out=0.5
                 ):
        """adapt first octconv , octconv and last octconv

        """
        assert alpha_in >= 0 and alpha_in <= 1, "the value of alpha_in should be in range of [0,1],but get {}".format(
            alpha_in)
        assert alpha_out >= 0 and alpha_out <= 1, "the value of alpha_in should be in range of [0,1],but get {}".format(
            alpha_out)
        super(OctConv2d_v1, self).__init__(in_channels,
                                        out_channels,
                                        dilation,
                                        groups,
                                        bias,)
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out
        self.kernel_size = (1,1)
        self.stride = (1,1)
        self.avgPool = nn.AvgPool2d(kernel_size, stride, padding)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.inChannelSplitIndex = int(
            self.alpha_in * self.in_channels)
        self.outChannelSplitIndex = int(
            self.alpha_out * self.out_channels)
        # split bias
        if bias:
            self.hh_bias = self.bias[self.outChannelSplitIndex:]
            self.hl_bias = self.bias[:self.outChannelSplitIndex]
            self.ll_bias = self.bias[ :self.outChannelSplitIndex]
            self.lh_bias = self.bias[ self.outChannelSplitIndex:]
        else:
            self.hh_bias = None
            self.hl_bias = None
            self.ll_bias = None
            self.ll_bias = None

        # conv and upsample
        self.upsample = F.interpolate

    def forward(self, x):
        if not isinstance(x, tuple):
            # first octconv
            input_h = x if self.alpha_in == 0 else None
            input_l = x if self.alpha_in == 1 else None
        else:
            input_l = x[0]
            input_h = x[1]

        output = [0, 0]
        # H->H
        if self.outChannelSplitIndex != self.out_channels and self.inChannelSplitIndex != self.in_channels:
            output_hh = F.conv2d(self.avgPool(input_h),
                                 self.weight[
                                 self.outChannelSplitIndex:,
                                 self.inChannelSplitIndex:,
                                 :, :],
                                 self.bias[self.outChannelSplitIndex:],
                                 self.kernel_size
                                 )

            output[1] += output_hh

        # H->L
        if self.outChannelSplitIndex != 0 and self.inChannelSplitIndex != self.in_channels:
            output_hl = F.conv2d(self.avgpool(self.avgPool(input_h)),
                                 self.weight[
                :self.outChannelSplitIndex,
                self.inChannelSplitIndex:,
                                     :, :],
                                 self.bias[:self.outChannelSplitIndex],
                                 self.kernel_size
                                 )

            output[0] += output_hl

        # L->L
        if self.outChannelSplitIndex != 0 and self.inChannelSplitIndex != 0:
            output_ll = F.conv2d((self.avgPool(input_l)),
                                 self.weight[
                                 :self.outChannelSplitIndex,
                                 :self.inChannelSplitIndex,
                                 :, :],
                                 self.bias[:self.outChannelSplitIndex],
                                 self.kernel_size
                                 )

            output[0] += output_ll

        # L->H
        if self.outChannelSplitIndex != self.out_channels and self.inChannelSplitIndex != 0:
            output_lh = F.conv2d(self.avgPool(input_l),
                                 self.weight[
                                 self.outChannelSplitIndex:,
                                 :self.inChannelSplitIndex,
                                 :, :],
                                 self.bias[self.outChannelSplitIndex:],
                                 self.kernel_size
                                 )
            output_lh = self.upsample(output_lh, scale_factor=2)

            output[1] += output_lh

        if isinstance(output[0], int):
            out = output[1]
        else:
            out = tuple(output)
        return out


class OctBN(nn.Module):
    def __init__(self, l_channels, h_channels):
        super(OctBN, self).__init__()
        # self.h_channels = h_channels
        # self.l_channels = l_channels

        self.h_BN = nn.BatchNorm2d(h_channels)
        self.l_BN = nn.BatchNorm2d(l_channels)

    def forward(self, x):
        out_l = self.l_BN(x[0])
        out_h = self.h_BN(x[1])

        return (out_l, out_h)


class OctAc(nn.Module):
    def __init__(self, name,inplace):
        super(OctAc, self).__init__()
        # self.h_channels = h_channels
        # self.l_channels = l_channels

        assert name in [
            "relu", "sigmoid","leakyrelu"], 'name should be in ["relu","sigmoid"] but get {} '.format(name)
        if name == "relu":
            self.ac = nn.ReLU(inplace)
        elif name == "leakyrelu":
            self.ac = nn.LeakyReLU(inplace)
        else:
            self.ac = nn.Sigmoid()

    def forward(self, x):
        out_l = self.ac(x[0])
        out_h = self.ac(x[1])

        return (out_l, out_h)
class OctDropout(nn.Module):
    def __init__(self,p):
        super(OctDropout,self).__init__()

        self.dropout = nn.Dropout(p)
    def forward(self, x):
        out_l = self.dropout(x[0])
        out_h = self.dropout(x[1])

        return (out_l,out_h)


if __name__ == "__main__":
    input = torch.randn(1, 3, 32, 32)
    octconv1 = OctConv2d(
        in_channels=3,
        out_channels=6,
        kernel_size=3,
        padding=1,
        stride=2,
        alpha_in=0,
        alpha_out=0.3)
    octconv2 = OctConv2d(
        in_channels=6,
        out_channels=16,
        kernel_size=2,
        padding=0,
        stride=2,
        alpha_in=0.3,
        alpha_out=0.5)
    lastconv = OctConv2d(
        in_channels=16,
        out_channels=32,
        kernel_size=2,
        padding=0,
        stride=2,
        alpha_in=0.5,
        alpha_out=0)
    # bn1 = OctBN(3,3)
    # ac1 = OctAc(name="relu")
    out = octconv1(input)
    print(len(out))
    print(out[0].size())
    print(out[1].size())
    out = octconv2(out)
    print(len(out))
    print(out[0].size())
    print(out[1].size())

    out = lastconv(out)
    print(len(out))
    print(out[0].size())
    print(out[1])
