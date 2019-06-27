# -*- coding: utf-8 -*-
# @Time    : 2019/4/23 10:30
# @Author  : ljf
from torch import nn
from models.OctConv_v1 import *
from models.OctConv_v2 import *
import argparse


def Conv2d(*args,is_octconv=False,octconv_type="",**kwargs):
    #print(*args)
    #print(kwargs)
    if not is_octconv:
        if "alpha_in" in kwargs.keys():
            kwargs.pop("alpha_in")
        if "alpha_out" in kwargs.keys():
            kwargs.pop("alpha_out")
        return nn.Conv2d(*args,**kwargs)
    elif is_octconv and octconv_type == "v1":
        return OctConv2d_v1(*args,**kwargs)
    elif is_octconv and octconv_type == "v2":
        return OctConv2d_v2(*args,**kwargs)
    else:
        print("you should be input is_octconv=False or is_octconv=True octconv_type = v1/v2, but get {}{}".format(is_octconv,octconv_type))
def BatchNorm2d(h_channels,l_channels,is_octconv=False):
    if is_octconv:
        return OctBN(h_channels,l_channels)
    else:
        return nn.BatchNorm2d(h_channels+l_channels)
def AC(name,inplace,is_octconv=False):
    """name: the name of activation function
    """
    if is_octconv:
        return OctAc(name,inplace)
    else:
        if name == "relu":
            return nn.ReLU(inplace)
        elif name == "sigmoid":
            return nn.Sigmoid(inplace)
        elif name == "leakyrelu":
            return nn.LeakyReLU(inplace)      

def Dropout(p,is_octconv=False):
    if is_octconv:
        return OctDropout(p)
    else:
        return nn.Dropout(p)
if __name__ == "__main__":
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')


    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--is_octconv', default="False", type=str2bool, metavar='oct',
                        help='whether use octconv')
    parser.add_argument('--octconv_type', default="", type=str,
                        metavar='oct_type', help='version of octconv,v1 and v2')
    # arg = {"is_octconv":False,"octconv_type":""}
    arg = parser.parse_args()
    i = 1
    j =3
    conv = Conv2d(1,3,kernel_size=3,padding=1,stride=2,is_octconv=True,octconv_type="")
    # conv(1,3,kernel_size=3,padding =1,stride=2)
    print(conv.in_channels)