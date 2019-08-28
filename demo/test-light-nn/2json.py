# -*- coding: utf-8 -*-
# @Time    : 2019/5/29 15:30
# @Author  : ljf
import json
import torch
from collections import OrderedDict
import jsonlib

# print (jsonlib.write (['Hello world!'], indent = '    ').decode ('utf8'))


def WriteResidualJson(write_path):
    json_dict = {
        "operators": [
            {
                "name": "conv1",
                "type": "Conv2D",
                "input": [
                    "input"
                ],
                "output": [
                    "conv1"
                ],
                "param": {
                    "input_size": 3,
                    "output_size": 16,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "bias": False
                }
            },  # conv1
            {
                "name": "bn1",
                "type": "BatchNorm2D",
                "input": [
                    "conv1"
                ],
                "output": [
                    "bn1"
                ],
                "param": {
                    "input_size": 16,
                    "output_size": 16
                }
            },  # bn1
            {
                "name": "leakyrelu",
                "type": "LeakyReLU",
                "input": [
                    "bn1"
                ],
                "output": [
                    "leakyrelu"
                ]
            },  # leakyrelu
            # layer1 block0
            {
                "name": "layer1.0.conv1",
                "type": "Conv2D",
                "input": [
                    "leakyrelu"
                ],
                "output": [
                    "layer1.0.conv1"
                ],
                "param": {
                    "input_size": 16,
                    "output_size": 16,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "bias": False
                }
            },  # conv1，输入leakyrelu
            {
                "name": "layer1.0.bn1",
                "type": "BatchNorm2D",
                "input": [
                    "layer1.0.conv1"
                ],
                "output": [
                    "layer1.0.bn1"
                ],
                "param": {
                    "input_size": 16,
                    "output_size": 16
                }
            },  # bn1
            {
                "name": "layer1.0.relu1",
                "type": "ReLU",
                "input": [
                    "layer1.0.bn1"
                ],
                "output": [
                    "layer1.0.relu1"
                ]
            },  # relu1
            {
                "name": "layer1.0.conv2",
                "type": "Conv2D",
                "input": [
                    "layer1.0.relu1"
                ],
                "output": [
                    "layer1.0.conv2"
                ],
                "param": {
                    "input_size": 16,
                    "output_size": 16,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "bias": False
                }
            },  # conv2
            {
                "name": "layer1.0.bn2",
                "type": "BatchNorm2D",
                "input": [
                    "layer1.0.conv2"
                ],
                "output": [
                    "layer1.0.bn2"
                ],
                "param": {
                    "input_size": 16,
                    "output_size": 16
                }
            },  # bn2
            {
                "name": "layer1.0.relu2",
                "type": "ReLU",
                "input": [
                    "layer1.0.bn2"
                ],
                "output": [
                    "layer1.0.relu2"
                ]
            },  # relu2,

            # layer1 block1
            {
                "name": "layer1.1.conv1",
                "type": "Conv2D",
                "input": [
                    "layer1.0.relu2"
                ],
                "output": [
                    "layer1.1.conv1"
                ],
                "param": {
                    "input_size": 16,
                    "output_size": 16,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "bias": False
                }
            },  # conv1，输入layer.0.relu2
            {
                "name": "layer1.1.bn1",
                "type": "BatchNorm2D",
                "input": [
                    "layer1.1.conv1"
                ],
                "output": [
                    "layer1.1.bn1"
                ],
                "param": {
                    "input_size": 16,
                    "output_size": 16
                }
            },  # bn1
            {
                "name": "layer1.1.relu1",
                "type": "ReLU",
                "input": [
                    "layer1.1.bn1"
                ],
                "output": [
                    "layer1.1.relu1"
                ]
            },  # relu1
            {
                "name": "layer1.1.conv2",
                "type": "Conv2D",
                "input": [
                    "layer1.1.relu1"
                ],
                "output": [
                    "layer1.1.conv2"
                ],
                "param": {
                    "input_size": 16,
                    "output_size": 16,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "bias": False
                }
            },  # conv2
            {
                "name": "layer1.1.bn2",
                "type": "BatchNorm2D",
                "input": [
                    "layer1.1.conv2"
                ],
                "output": [
                    "layer1.1.bn2"
                ],
                "param": {
                    "input_size": 16,
                    "output_size": 16
                }
            },  # bn2
            {
                "name": "layer1.1.relu2",
                "type": "ReLU",
                "input": [
                    "layer1.1.bn2"
                ],
                "output": [
                    "layer1.1.relu2"
                ]
            },  # relu2,
            # layer1 block2
            {
                "name": "layer1.2.conv1",
                "type": "Conv2D",
                "input": [
                    "layer1.1.relu2"
                ],
                "output": [
                    "layer1.2.conv1"
                ],
                "param": {
                    "input_size": 16,
                    "output_size": 16,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "bias": False
                }
            },  # conv1，输入layer.0.relu2
            {
                "name": "layer1.2.bn1",
                "type": "BatchNorm2D",
                "input": [
                    "layer1.2.conv1"
                ],
                "output": [
                    "layer1.2.bn1"
                ],
                "param": {
                    "input_size": 16,
                    "output_size": 16
                }
            },  # bn1
            {
                "name": "layer1.2.relu1",
                "type": "ReLU",
                "input": [
                    "layer1.2.bn1"
                ],
                "output": [
                    "layer1.2.relu1"
                ]
            },  # relu1
            {
                "name": "layer1.2.conv2",
                "type": "Conv2D",
                "input": [
                    "layer1.2.relu1"
                ],
                "output": [
                    "layer1.2.conv2"
                ],
                "param": {
                    "input_size": 16,
                    "output_size": 16,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "bias": False
                }
            },  # conv2
            {
                "name": "layer1.2.bn2",
                "type": "BatchNorm2D",
                "input": [
                    "layer1.2.conv2"
                ],
                "output": [
                    "layer1.2.bn2"
                ],
                "param": {
                    "input_size": 16,
                    "output_size": 16
                }
            },  # bn2
            {
                "name": "layer1.2.relu2",
                "type": "ReLU",
                "input": [
                    "layer1.2.bn2"
                ],
                "output": [
                    "layer1.2.relu2"
                ]
            },  # relu2,



        ]
    }
    # print (json.dumps({'a': 'Runoob', 'b': 7}, sort_keys=True, indent=4, separators=(',', ': ')))
    with open(write_path, "w") as file:
        json.dump(json_dict, file, indent=4, separators=(',', ': '))

def WriteResnet20(write_path):
    """
    ['conv1', 'bn1',

     'layer1.0.conv1', 'layer1.0.bn1', 'layer1.0.conv2', 'layer1.0.bn2',
      'layer1.1.conv1', 'layer1.1.bn1', 'layer1.1.conv2', 'layer1.1.bn2',
      'layer1.2.conv1', 'layer1.2.bn1', 'layer1.2.conv2', 'layer1.2.bn2',
    :param write_path:
    :return:
    """
    json_dict = {
        "operators": [
            {
                "name": "conv1",
                "type": "Conv2D",
                "input": [
                    "input"
                ],
                "output": [
                    "conv1"
                ],
                "param": {
                    "input_size": 3,
                    "output_size": 16,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "bias": False
                }
            },  # conv1
            {
                "name": "bn1",
                "type": "BatchNorm2D",
                "input": [
                    "conv1"
                ],
                "output": [
                    "bn1"
                ],
                "param": {
                    "input_size": 16,
                    "output_size": 16
                }
            },  # bn1
            {
                "name": "leakyrelu",
                "type": "LeakyReLU",
                "input": [
                    "bn1"
                ],
                "output": [
                    "leakyrelu"
                ]
            },  # leakyrelu
            # layer1 block0
            {
                "name": "layer1.0.conv1",
                "type": "Conv2D",
                "input": [
                    "leakyrelu"
                ],
                "output": [
                    "layer1.0.conv1"
                ],
                "param": {
                    "input_size": 16,
                    "output_size": 16,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "bias": False
                }
            },  # conv1，输入leakyrelu
            {
                "name": "layer1.0.bn1",
                "type": "BatchNorm2D",
                "input": [
                    "layer1.0.conv1"
                ],
                "output": [
                    "layer1.0.bn1"
                ],
                "param": {
                    "input_size": 16,
                    "output_size": 16
                }
            },  # bn1
            {
                "name": "layer1.0.relu1",
                "type": "ReLU",
                "input": [
                    "layer1.0.bn1"
                ],
                "output": [
                    "layer1.0.relu1"
                ]
            },  # relu1
            {
                "name": "layer1.0.conv2",
                "type": "Conv2D",
                "input": [
                    "layer1.0.relu1"
                ],
                "output": [
                    "layer1.0.conv2"
                ],
                "param": {
                    "input_size": 16,
                    "output_size": 16,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "bias": False
                }
            },  # conv2
            {
                "name": "layer1.0.bn2",
                "type": "BatchNorm2D",
                "input": [
                    "layer1.0.conv2"
                ],
                "output": [
                    "layer1.0.bn2"
                ],
                "param": {
                    "input_size": 16,
                    "output_size": 16
                }
            },  # bn2
            {
                "name": "layer1.0.residual",
                "type": "Residual",
                "input": [
                    "layer1.0.bn2",
                    "leakyrelu"
                ],
                "output": [
                    "layer1.0.residual"
                ]
            },  # residual
            {
                "name": "layer1.0.relu2",
                "type": "ReLU",
                "input": [
                    "layer1.0.residual"
                ],
                "output": [
                    "layer1.0.relu2"
                ]
            },  # relu2,

            # layer1 block1
            {
                "name": "layer1.1.conv1",
                "type": "Conv2D",
                "input": [
                    "layer1.0.relu2"
                ],
                "output": [
                    "layer1.1.conv1"
                ],
                "param": {
                    "input_size": 16,
                    "output_size": 16,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "bias": False
                }
            },  # conv1，输入layer.0.relu2
            {
                "name": "layer1.1.bn1",
                "type": "BatchNorm2D",
                "input": [
                    "layer1.1.conv1"
                ],
                "output": [
                    "layer1.1.bn1"
                ],
                "param": {
                    "input_size": 16,
                    "output_size": 16
                }
            },  # bn1
            {
                "name": "layer1.1.relu1",
                "type": "ReLU",
                "input": [
                    "layer1.1.bn1"
                ],
                "output": [
                    "layer1.1.relu1"
                ]
            },  # relu1
            {
                "name": "layer1.1.conv2",
                "type": "Conv2D",
                "input": [
                    "layer1.1.relu1"
                ],
                "output": [
                    "layer1.1.conv2"
                ],
                "param": {
                    "input_size": 16,
                    "output_size": 16,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "bias": False
                }
            },  # conv2
            {
                "name": "layer1.1.bn2",
                "type": "BatchNorm2D",
                "input": [
                    "layer1.1.conv2"
                ],
                "output": [
                    "layer1.1.bn2"
                ],
                "param": {
                    "input_size": 16,
                    "output_size": 16
                }
            },  # bn2
            {
                "name": "layer1.1.residual",
                "type": "Residual",
                "input": [
                    "layer1.1.bn2",
                    "layer1.0.relu2"
                ],
                "output": [
                    "layer1.1.residual"
                ]
            },  # residual
            {
                "name": "layer1.1.relu2",
                "type": "ReLU",
                "input": [
                    "layer1.1.residual"
                ],
                "output": [
                    "layer1.1.relu2"
                ]
            },  # relu2,
            # layer1 block2
            {
                "name": "layer1.2.conv1",
                "type": "Conv2D",
                "input": [
                    "layer1.1.relu2"
                ],
                "output": [
                    "layer1.2.conv1"
                ],
                "param": {
                    "input_size": 16,
                    "output_size": 16,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "bias": False
                }
            },  # conv1，输入layer.0.relu2
            {
                "name": "layer1.2.bn1",
                "type": "BatchNorm2D",
                "input": [
                    "layer1.2.conv1"
                ],
                "output": [
                    "layer1.2.bn1"
                ],
                "param": {
                    "input_size": 16,
                    "output_size": 16
                }
            },  # bn1
            {
                "name": "layer1.2.relu1",
                "type": "ReLU",
                "input": [
                    "layer1.2.bn1"
                ],
                "output": [
                    "layer1.2.relu1"
                ]
            },  # relu1
            {
                "name": "layer1.2.conv2",
                "type": "Conv2D",
                "input": [
                    "layer1.2.relu1"
                ],
                "output": [
                    "layer1.2.conv2"
                ],
                "param": {
                    "input_size": 16,
                    "output_size": 16,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "bias": False
                }
            },  # conv2
            {
                "name": "layer1.2.bn2",
                "type": "BatchNorm2D",
                "input": [
                    "layer1.2.conv2"
                ],
                "output": [
                    "layer1.2.bn2"
                ],
                "param": {
                    "input_size": 16,
                    "output_size": 16
                }
            },  # bn2
            {
                "name": "layer1.2.residual",
                "type": "Residual",
                "input": [
                    "layer1.2.bn2",
                    "layer1.1.relu2"
                ],
                "output": [
                    "layer1.2.residual"
                ]
            },  # residual
            {
                "name": "layer1.2.relu2",
                "type": "ReLU",
                "input": [
                    "layer1.2.residual"
                ],
                "output": [
                    "layer1.2.relu2"
                ]
            },  # relu2,

            # layer2 block0
            #'layer2.0.conv1', 'layer2.0.bn1', 'layer2.0.conv2', 'layer2.0.bn2', 'layer2.0.downsample.0', 'layer2.0.downsample.1',残差链接
            {
                "name": "layer2.0.conv1",
                "type": "Conv2D",
                "input": [
                    "layer1.2.relu2"
                ],
                "output": [
                    "layer2.0.conv1"
                ],
                "param": {
                    "input_size": 16,
                    "output_size": 32,
                    "kernel_size": 3,
                    "stride": 2,
                    "padding": 1,
                    "bias": False
                }
            },  # conv1，输入layer1.2.relu2
            {
                "name": "layer2.0.bn1",
                "type": "BatchNorm2D",
                "input": [
                    "layer2.0.conv1"
                ],
                "output": [
                    "layer2.0.bn1"
                ],
                "param": {
                    "input_size": 32,
                    "output_size": 32
                }
            },  # bn1
            {
                "name": "layer2.0.relu1",
                "type": "ReLU",
                "input": [
                    "layer2.0.bn1"
                ],
                "output": [
                    "layer2.0.relu1"
                ]
            },  # relu1
            {
                "name": "layer2.0.conv2",
                "type": "Conv2D",
                "input": [
                    "layer2.0.relu1"
                ],
                "output": [
                    "layer2.0.conv2"
                ],
                "param": {
                    "input_size": 32,
                    "output_size": 32,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "bias": False
                }
            },  # conv2
            {
                "name": "layer2.0.bn2",
                "type": "BatchNorm2D",
                "input": [
                    "layer2.0.conv2"
                ],
                "output": [
                    "layer2.0.bn2"
                ],
                "param": {
                    "input_size": 32,
                    "output_size": 32
                }
            },  # bn2
            # downsample
            {
                "name": "layer2.0.downsample.0",
                "type": "Conv2D",
                "input": [
                    "layer1.2.relu2"
                ],
                "output": [
                    "layer2.0.downsample.0"
                ],
                "param": {
                    "input_size": 16,
                    "output_size": 32,
                    "kernel_size": 1,
                    "stride": 2,
                    "padding": 0,
                    "bias": False
                }
            },  # NIN
            {
                "name": "layer2.0.downsample.1",
                "type": "BatchNorm2D",
                "input": [
                    "layer2.0.downsample.0"
                ],
                "output": [
                    "layer2.0.downsample.1"
                ],
                "param": {
                    "input_size": 32,
                    "output_size": 32
                }
            },  # bn
            {
                "name": "layer2.0.residual",
                "type": "Residual",
                "input": [
                    "layer2.0.bn2",
                    "layer2.0.downsample.1"
                ],
                "output": [
                    "layer2.0.residual"
                ]
            },# residual
            {
                "name": "layer2.0.relu2",
                "type": "ReLU",
                "input": [
                    "layer2.0.residual"
                ],
                "output": [
                    "layer2.0.relu2"
                ]
            },  # relu2,

            # layer2 block1
            # 'layer2.1.conv1', 'layer2.1.bn1', 'layer2.1.conv2', 'layer2.1.bn2',
            {
                "name": "layer2.1.conv1",
                "type": "Conv2D",
                "input": [
                    "layer2.0.relu2"
                ],
                "output": [
                    "layer2.1.conv1"
                ],
                "param": {
                    "input_size": 32,
                    "output_size": 32,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "bias": False
                }
            },  # conv1，输入layer2.1.relu2
            {
                "name": "layer2.1.bn1",
                "type": "BatchNorm2D",
                "input": [
                    "layer2.1.conv1"
                ],
                "output": [
                    "layer2.1.bn1"
                ],
                "param": {
                    "input_size": 32,
                    "output_size": 32
                }
            },  # bn1
            {
                "name": "layer2.1.relu1",
                "type": "ReLU",
                "input": [
                    "layer2.1.bn1"
                ],
                "output": [
                    "layer2.1.relu1"
                ]
            },  # relu1
            {
                "name": "layer2.1.conv2",
                "type": "Conv2D",
                "input": [
                    "layer2.1.relu1"
                ],
                "output": [
                    "layer2.1.conv2"
                ],
                "param": {
                    "input_size": 32,
                    "output_size": 32,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "bias": False
                }
            },  # conv2
            {
                "name": "layer2.1.bn2",
                "type": "BatchNorm2D",
                "input": [
                    "layer2.1.conv2"
                ],
                "output": [
                    "layer2.1.bn2"
                ],
                "param": {
                    "input_size": 32,
                    "output_size": 32
                }
            },  # bn2
            {
                "name": "layer2.1.residual",
                "type": "Residual",
                "input": [
                    "layer2.1.bn2",
                    "layer2.0.relu2"
                ],
                "output": [
                    "layer2.1.residual"
                ]
            },  # residual
            {
                "name": "layer2.1.relu2",
                "type": "ReLU",
                "input": [
                    "layer2.1.residual"
                ],
                "output": [
                    "layer2.1.relu2"
                ]
            },  # relu2,

            # layer2 block2
            # 'layer2.2.conv1', 'layer2.2.bn1', 'layer2.2.conv2', 'layer2.2.bn2',
            {
                "name": "layer2.2.conv1",
                "type": "Conv2D",
                "input": [
                    "layer2.1.relu2"
                ],
                "output": [
                    "layer2.2.conv1"
                ],
                "param": {
                    "input_size": 32,
                    "output_size": 32,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "bias": False
                }
            },  # conv1，输入layer2.1.relu2
            {
                "name": "layer2.2.bn1",
                "type": "BatchNorm2D",
                "input": [
                    "layer2.2.conv1"
                ],
                "output": [
                    "layer2.2.bn1"
                ],
                "param": {
                    "input_size": 32,
                    "output_size": 32
                }
            },  # bn1
            {
                "name": "layer2.2.relu1",
                "type": "ReLU",
                "input": [
                    "layer2.2.bn1"
                ],
                "output": [
                    "layer2.2.relu1"
                ]
            },  # relu1
            {
                "name": "layer2.2.conv2",
                "type": "Conv2D",
                "input": [
                    "layer2.2.relu1"
                ],
                "output": [
                    "layer2.2.conv2"
                ],
                "param": {
                    "input_size": 32,
                    "output_size": 32,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "bias": False
                }
            },  # conv2
            {
                "name": "layer2.2.bn2",
                "type": "BatchNorm2D",
                "input": [
                    "layer2.2.conv2"
                ],
                "output": [
                    "layer2.2.bn2"
                ],
                "param": {
                    "input_size": 32,
                    "output_size": 32
                }
            },  # bn2
            {
                "name": "layer2.2.residual",
                "type": "Residual",
                "input": [
                    "layer2.2.bn2",
                    "layer2.1.relu2"
                ],
                "output": [
                    "layer2.2.residual"
                ]
            },  # residual
            {
                "name": "layer2.2.relu2",
                "type": "ReLU",
                "input": [
                    "layer2.2.residual"
                ],
                "output": [
                    "layer2.2.relu2"
                ]
            },  # relu2,

            # layer3 block0
            #'layer3.0.conv1', 'layer3.0.bn1', 'layer3.0.conv2', 'layer3.0.bn2', 'layer3.0.downsample.0', 'layer3.0.downsample.1', 残差链接
            {
                "name": "layer3.0.conv1",
                "type": "Conv2D",
                "input": [
                    "layer2.2.relu2"
                ],
                "output": [
                    "layer3.0.conv1"
                ],
                "param": {
                    "input_size": 32,
                    "output_size": 64,
                    "kernel_size": 3,
                    "stride": 2,
                    "padding": 1,
                    "bias": False
                }
            },  # conv1，输入layer2.2.relu2
            {
                "name": "layer3.0.bn1",
                "type": "BatchNorm2D",
                "input": [
                    "layer3.0.conv1"
                ],
                "output": [
                    "layer3.0.bn1"
                ],
                "param": {
                    "input_size": 64,
                    "output_size": 64
                }
            },  # bn1
            {
                "name": "layer3.0.relu1",
                "type": "ReLU",
                "input": [
                    "layer3.0.bn1"
                ],
                "output": [
                    "layer3.0.relu1"
                ]
            },  # relu1
            {
                "name": "layer3.0.conv2",
                "type": "Conv2D",
                "input": [
                    "layer3.0.relu1"
                ],
                "output": [
                    "layer3.0.conv2"
                ],
                "param": {
                    "input_size": 64,
                    "output_size": 64,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "bias": False
                }
            },  # conv2
            {
                "name": "layer3.0.bn2",
                "type": "BatchNorm2D",
                "input": [
                    "layer3.0.conv2"
                ],
                "output": [
                    "layer3.0.bn2"
                ],
                "param": {
                    "input_size": 64,
                    "output_size": 64
                }
            },  # bn2
            # downsample
            {
                "name": "layer3.0.downsample.0",
                "type": "Conv2D",
                "input": [
                    "layer2.2.relu2"
                ],
                "output": [
                    "layer3.0.downsample.0"
                ],
                "param": {
                    "input_size": 32,
                    "output_size": 64,
                    "kernel_size": 1,
                    "stride": 2,
                    "padding": 0,
                    "bias": False
                }
            },  # NIN
            {
                "name": "layer3.0.downsample.1",
                "type": "BatchNorm2D",
                "input": [
                    "layer3.0.downsample.0"
                ],
                "output": [
                    "layer3.0.downsample.1"
                ],
                "param": {
                    "input_size": 64,
                    "output_size": 64
                }
            },  # bn
            {
                "name": "layer3.0.residual",
                "type": "Residual",
                "input": [
                    "layer3.0.bn2",
                    "layer3.0.downsample.1"
                ],
                "output": [
                    "layer3.0.residual"
                ]
            },  # residual
            {
                "name": "layer3.0.relu2",
                "type": "ReLU",
                "input": [
                    "layer3.0.residual"
                ],
                "output": [
                    "layer3.0.relu2"
                ]
            },  # relu2,
            # layer3 block1
            # 'layer3.1.conv1', 'layer3.1.bn1', 'layer3.1.conv2', 'layer3.1.bn2',
            {
                "name": "layer3.1.conv1",
                "type": "Conv2D",
                "input": [
                    "layer3.0.relu2"
                ],
                "output": [
                    "layer3.1.conv1"
                ],
                "param": {
                    "input_size": 64,
                    "output_size": 64,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "bias": False
                }
            },  # conv1，输入layer3.0.relu2
            {
                "name": "layer3.1.bn1",
                "type": "BatchNorm2D",
                "input": [
                    "layer3.1.conv1"
                ],
                "output": [
                    "layer3.1.bn1"
                ],
                "param": {
                    "input_size": 64,
                    "output_size": 64
                }
            },  # bn1
            {
                "name": "layer3.1.relu1",
                "type": "ReLU",
                "input": [
                    "layer3.1.bn1"
                ],
                "output": [
                    "layer3.1.relu1"
                ]
            },  # relu1
            {
                "name": "layer3.1.conv2",
                "type": "Conv2D",
                "input": [
                    "layer3.1.relu1"
                ],
                "output": [
                    "layer3.1.conv2"
                ],
                "param": {
                    "input_size": 64,
                    "output_size": 64,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "bias": False
                }
            },  # conv2
            {
                "name": "layer3.1.bn2",
                "type": "BatchNorm2D",
                "input": [
                    "layer3.1.conv2"
                ],
                "output": [
                    "layer3.1.bn2"
                ],
                "param": {
                    "input_size": 64,
                    "output_size": 64
                }
            },  # bn2
            {
                "name": "layer3.1.residual",
                "type": "Residual",
                "input": [
                    "layer3.1.bn2",
                    "layer3.0.relu2"
                ],
                "output": [
                    "layer3.1.residual"
                ]
            },  # residual
            {
                "name": "layer3.1.relu2",
                "type": "ReLU",
                "input": [
                    "layer3.1.residual"
                ],
                "output": [
                    "layer3.1.relu2"
                ]
            },  # relu2,

           # layer3 block2
           # 'layer3.2.conv1', 'layer3.2.bn1', 'layer3.2.conv2', 'layer3.2.bn2',
            {
                "name": "layer3.2.conv1",
                "type": "Conv2D",
                "input": [
                    "layer3.1.relu2"
                ],
                "output": [
                    "layer3.2.conv1"
                ],
                "param": {
                    "input_size": 64,
                    "output_size": 64,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "bias": False
                }
            },  # conv1，输入layer2.1.relu2
            {
                "name": "layer3.2.bn1",
                "type": "BatchNorm2D",
                "input": [
                    "layer3.2.conv1"
                ],
                "output": [
                    "layer3.2.bn1"
                ],
                "param": {
                    "input_size": 64,
                    "output_size": 64
                }
            },  # bn1
            {
                "name": "layer3.2.relu1",
                "type": "ReLU",
                "input": [
                    "layer3.2.bn1"
                ],
                "output": [
                    "layer3.2.relu1"
                ]
            },  # relu1
            {
                "name": "layer3.2.conv2",
                "type": "Conv2D",
                "input": [
                    "layer3.2.relu1"
                ],
                "output": [
                    "layer3.2.conv2"
                ],
                "param": {
                    "input_size": 64,
                    "output_size": 64,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "bias": False
                }
            },  # conv2
            {
                "name": "layer3.2.bn2",
                "type": "BatchNorm2D",
                "input": [
                    "layer3.2.conv2"
                ],
                "output": [
                    "layer3.2.bn2"
                ],
                "param": {
                    "input_size": 64,
                    "output_size": 64
                }
            },  # bn2
            {
                "name": "layer3.2.residual",
                "type": "Residual",
                "input": [
                    "layer3.2.bn2",
                    "layer3.1.relu2"
                ],
                "output": [
                    "layer3.2.residual"
                ]
            },  # residual
            {
                "name": "layer3.2.relu2",
                "type": "ReLU",
                "input": [
                    "layer3.2.residual"
                ],
                "output": [
                    "layer3.2.relu2"
                ]
            },  # relu2,
           # maxpool 'bn2', 'fc'
            {
                "name": "pooling2d",
                "type": "Pooling2D",
                "input": [
                    "layer3.2.relu2"
                ],
                "output": [
                    "pooling2d"
                ],
                "param": {
                    "input_size": 64,
                    "output_size": 64,
                    "global_pooling":True,
                    "padding": 0
                }
            },
            {
                "name": "bn2",
                "type": "BatchNorm2D",
                "input": [
                    "pooling2d"
                ],
                "output": [
                    "bn2"
                ],
                "param": {
                    "input_size": 64,
                    "output_size": 64
                }
            },  # bn2
            {
                "name": "fc",
                "type": "Linear",
                "input": [
                    "bn2"
                ],
                "output": [
                    "fc"
                ],
                "param": {
                    "input_size": 64,
                    "output_size": 7,
                    "bias": True
                }
            },
        ]
    }
    # print (json.dumps({'a': 'Runoob', 'b': 7}, sort_keys=True, indent=4, separators=(',', ': ')))
    with open(write_path, "w") as file:
        json.dump(json_dict, file, indent=4, separators=(',', ': '))

def WriteResnet14(write_path):
    """
    ['conv1', 'bn1',

     'layer1.0.conv1', 'layer1.0.bn1', 'layer1.0.conv2', 'layer1.0.bn2',
      'layer1.1.conv1', 'layer1.1.bn1', 'layer1.1.conv2', 'layer1.1.bn2',
      'layer1.2.conv1', 'layer1.2.bn1', 'layer1.2.conv2', 'layer1.2.bn2',
    :param write_path:
    :return:
    """
    json_dict = {
        "operators": [
            {
                "name": "conv1",
                "type": "Conv2D",
                "input": [
                    "input"
                ],
                "output": [
                    "conv1"
                ],
                "param": {
                    "input_size": 3,
                    "output_size": 16,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "bias": False
                }
            },  # conv1
            {
                "name": "bn1",
                "type": "BatchNorm2D",
                "input": [
                    "conv1"
                ],
                "output": [
                    "bn1"
                ],
                "param": {
                    "input_size": 16,
                    "output_size": 16
                }
            },  # bn1
            {
                "name": "leakyrelu",
                "type": "LeakyReLU",
                "input": [
                    "bn1"
                ],
                "output": [
                    "leakyrelu"
                ]
            },  # leakyrelu
            # layer1 block0
            {
                "name": "layer1.0.conv1",
                "type": "Conv2D",
                "input": [
                    "leakyrelu"
                ],
                "output": [
                    "layer1.0.conv1"
                ],
                "param": {
                    "input_size": 16,
                    "output_size": 16,
                    "kernel_size": 3,
                    "stride": 2,
                    "padding": 1,
                    "bias": False
                }
            },  # conv1，输入leakyrelu
            {
                "name": "layer1.0.bn1",
                "type": "BatchNorm2D",
                "input": [
                    "layer1.0.conv1"
                ],
                "output": [
                    "layer1.0.bn1"
                ],
                "param": {
                    "input_size": 16,
                    "output_size": 16
                }
            },  # bn1
            {
                "name": "layer1.0.relu1",
                "type": "ReLU",
                "input": [
                    "layer1.0.bn1"
                ],
                "output": [
                    "layer1.0.relu1"
                ]
            },  # relu1
            {
                "name": "layer1.0.conv2",
                "type": "Conv2D",
                "input": [
                    "layer1.0.relu1"
                ],
                "output": [
                    "layer1.0.conv2"
                ],
                "param": {
                    "input_size": 16,
                    "output_size": 16,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "bias": False
                }
            },  # conv2
            {
                "name": "layer1.0.bn2",
                "type": "BatchNorm2D",
                "input": [
                    "layer1.0.conv2"
                ],
                "output": [
                    "layer1.0.bn2"
                ],
                "param": {
                    "input_size": 16,
                    "output_size": 16
                }
            },  # bn2
            # downsample
            {
                "name": "layer1.0.downsample.0",
                "type": "Conv2D",
                "input": [
                    "leakyrelu"
                ],
                "output": [
                    "layer1.0.downsample.0"
                ],
                "param": {
                    "input_size": 16,
                    "output_size": 16,
                    "kernel_size": 1,
                    "stride": 2,
                    "padding": 0,
                    "bias": False
                }
            },  # NIN
            {
                "name": "layer1.0.downsample.1",
                "type": "BatchNorm2D",
                "input": [
                    "layer1.0.downsample.0"
                ],
                "output": [
                    "layer1.0.downsample.1"
                ],
                "param": {
                    "input_size": 16,
                    "output_size": 16
                }
            },  # bn
            {
                "name": "layer1.0.residual",
                "type": "Residual",
                "input": [
                    "layer1.0.bn2",
                    "layer1.0.downsample.1"
                ],
                "output": [
                    "layer1.0.residual"
                ]
            },  # residual
            {
                "name": "layer1.0.relu2",
                "type": "ReLU",
                "input": [
                    "layer1.0.residual"
                ],
                "output": [
                    "layer1.0.relu2"
                ]
            },  # relu2,

            # layer1 block1
            {
                "name": "layer1.1.conv1",
                "type": "Conv2D",
                "input": [
                    "layer1.0.relu2"
                ],
                "output": [
                    "layer1.1.conv1"
                ],
                "param": {
                    "input_size": 16,
                    "output_size": 16,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "bias": False
                }
            },  # conv1，输入layer.0.relu2
            {
                "name": "layer1.1.bn1",
                "type": "BatchNorm2D",
                "input": [
                    "layer1.1.conv1"
                ],
                "output": [
                    "layer1.1.bn1"
                ],
                "param": {
                    "input_size": 16,
                    "output_size": 16
                }
            },  # bn1
            {
                "name": "layer1.1.relu1",
                "type": "ReLU",
                "input": [
                    "layer1.1.bn1"
                ],
                "output": [
                    "layer1.1.relu1"
                ]
            },  # relu1
            {
                "name": "layer1.1.conv2",
                "type": "Conv2D",
                "input": [
                    "layer1.1.relu1"
                ],
                "output": [
                    "layer1.1.conv2"
                ],
                "param": {
                    "input_size": 16,
                    "output_size": 16,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "bias": False
                }
            },  # conv2
            {
                "name": "layer1.1.bn2",
                "type": "BatchNorm2D",
                "input": [
                    "layer1.1.conv2"
                ],
                "output": [
                    "layer1.1.bn2"
                ],
                "param": {
                    "input_size": 16,
                    "output_size": 16
                }
            },  # bn2
            {
                "name": "layer1.1.residual",
                "type": "Residual",
                "input": [
                    "layer1.1.bn2",
                    "layer1.0.relu2"
                ],
                "output": [
                    "layer1.1.residual"
                ]
            },  # residual
            {
                "name": "layer1.1.relu2",
                "type": "ReLU",
                "input": [
                    "layer1.1.residual"
                ],
                "output": [
                    "layer1.1.relu2"
                ]
            },  # relu2,
            # layer1 block2
            {
                "name": "layer1.2.conv1",
                "type": "Conv2D",
                "input": [
                    "layer1.1.relu2"
                ],
                "output": [
                    "layer1.2.conv1"
                ],
                "param": {
                    "input_size": 16,
                    "output_size": 16,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "bias": False
                }
            },  # conv1，输入layer.0.relu2
            {
                "name": "layer1.2.bn1",
                "type": "BatchNorm2D",
                "input": [
                    "layer1.2.conv1"
                ],
                "output": [
                    "layer1.2.bn1"
                ],
                "param": {
                    "input_size": 16,
                    "output_size": 16
                }
            },  # bn1
            {
                "name": "layer1.2.relu1",
                "type": "ReLU",
                "input": [
                    "layer1.2.bn1"
                ],
                "output": [
                    "layer1.2.relu1"
                ]
            },  # relu1
            {
                "name": "layer1.2.conv2",
                "type": "Conv2D",
                "input": [
                    "layer1.2.relu1"
                ],
                "output": [
                    "layer1.2.conv2"
                ],
                "param": {
                    "input_size": 16,
                    "output_size": 16,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "bias": False
                }
            },  # conv2
            {
                "name": "layer1.2.bn2",
                "type": "BatchNorm2D",
                "input": [
                    "layer1.2.conv2"
                ],
                "output": [
                    "layer1.2.bn2"
                ],
                "param": {
                    "input_size": 16,
                    "output_size": 16
                }
            },  # bn2
            {
                "name": "layer1.2.residual",
                "type": "Residual",
                "input": [
                    "layer1.2.bn2",
                    "layer1.1.relu2"
                ],
                "output": [
                    "layer1.2.residual"
                ]
            },  # residual
            {
                "name": "layer1.2.relu2",
                "type": "ReLU",
                "input": [
                    "layer1.2.residual"
                ],
                "output": [
                    "layer1.2.relu2"
                ]
            },  # relu2,

            # layer2 block0
            #'layer2.0.conv1', 'layer2.0.bn1', 'layer2.0.conv2', 'layer2.0.bn2', 'layer2.0.downsample.0', 'layer2.0.downsample.1',残差链接
            {
                "name": "layer2.0.conv1",
                "type": "Conv2D",
                "input": [
                    "layer1.2.relu2"
                ],
                "output": [
                    "layer2.0.conv1"
                ],
                "param": {
                    "input_size": 16,
                    "output_size": 32,
                    "kernel_size": 3,
                    "stride": 2,
                    "padding": 1,
                    "bias": False
                }
            },  # conv1，输入layer1.2.relu2
            {
                "name": "layer2.0.bn1",
                "type": "BatchNorm2D",
                "input": [
                    "layer2.0.conv1"
                ],
                "output": [
                    "layer2.0.bn1"
                ],
                "param": {
                    "input_size": 32,
                    "output_size": 32
                }
            },  # bn1
            {
                "name": "layer2.0.relu1",
                "type": "ReLU",
                "input": [
                    "layer2.0.bn1"
                ],
                "output": [
                    "layer2.0.relu1"
                ]
            },  # relu1
            {
                "name": "layer2.0.conv2",
                "type": "Conv2D",
                "input": [
                    "layer2.0.relu1"
                ],
                "output": [
                    "layer2.0.conv2"
                ],
                "param": {
                    "input_size": 32,
                    "output_size": 32,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "bias": False
                }
            },  # conv2
            {
                "name": "layer2.0.bn2",
                "type": "BatchNorm2D",
                "input": [
                    "layer2.0.conv2"
                ],
                "output": [
                    "layer2.0.bn2"
                ],
                "param": {
                    "input_size": 32,
                    "output_size": 32
                }
            },  # bn2
            # downsample
            {
                "name": "layer2.0.downsample.0",
                "type": "Conv2D",
                "input": [
                    "layer1.2.relu2"
                ],
                "output": [
                    "layer2.0.downsample.0"
                ],
                "param": {
                    "input_size": 16,
                    "output_size": 32,
                    "kernel_size": 1,
                    "stride": 2,
                    "padding": 0,
                    "bias": False
                }
            },  # NIN
            {
                "name": "layer2.0.downsample.1",
                "type": "BatchNorm2D",
                "input": [
                    "layer2.0.downsample.0"
                ],
                "output": [
                    "layer2.0.downsample.1"
                ],
                "param": {
                    "input_size": 32,
                    "output_size": 32
                }
            },  # bn
            {
                "name": "layer2.0.residual",
                "type": "Residual",
                "input": [
                    "layer2.0.bn2",
                    "layer2.0.downsample.1"
                ],
                "output": [
                    "layer2.0.residual"
                ]
            },# residual
            {
                "name": "layer2.0.relu2",
                "type": "ReLU",
                "input": [
                    "layer2.0.residual"
                ],
                "output": [
                    "layer2.0.relu2"
                ]
            },  # relu2,

            # layer2 block1
            # 'layer2.1.conv1', 'layer2.1.bn1', 'layer2.1.conv2', 'layer2.1.bn2',
            {
                "name": "layer2.1.conv1",
                "type": "Conv2D",
                "input": [
                    "layer2.0.relu2"
                ],
                "output": [
                    "layer2.1.conv1"
                ],
                "param": {
                    "input_size": 32,
                    "output_size": 32,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "bias": False
                }
            },  # conv1，输入layer2.1.relu2
            {
                "name": "layer2.1.bn1",
                "type": "BatchNorm2D",
                "input": [
                    "layer2.1.conv1"
                ],
                "output": [
                    "layer2.1.bn1"
                ],
                "param": {
                    "input_size": 32,
                    "output_size": 32
                }
            },  # bn1
            {
                "name": "layer2.1.relu1",
                "type": "ReLU",
                "input": [
                    "layer2.1.bn1"
                ],
                "output": [
                    "layer2.1.relu1"
                ]
            },  # relu1
            {
                "name": "layer2.1.conv2",
                "type": "Conv2D",
                "input": [
                    "layer2.1.relu1"
                ],
                "output": [
                    "layer2.1.conv2"
                ],
                "param": {
                    "input_size": 32,
                    "output_size": 32,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "bias": False
                }
            },  # conv2
            {
                "name": "layer2.1.bn2",
                "type": "BatchNorm2D",
                "input": [
                    "layer2.1.conv2"
                ],
                "output": [
                    "layer2.1.bn2"
                ],
                "param": {
                    "input_size": 32,
                    "output_size": 32
                }
            },  # bn2
            {
                "name": "layer2.1.residual",
                "type": "Residual",
                "input": [
                    "layer2.1.bn2",
                    "layer2.0.relu2"
                ],
                "output": [
                    "layer2.1.residual"
                ]
            },  # residual
            {
                "name": "layer2.1.relu2",
                "type": "ReLU",
                "input": [
                    "layer2.1.residual"
                ],
                "output": [
                    "layer2.1.relu2"
                ]
            },  # relu2,

            # layer2 block2
            # 'layer2.2.conv1', 'layer2.2.bn1', 'layer2.2.conv2', 'layer2.2.bn2',
            {
                "name": "layer2.2.conv1",
                "type": "Conv2D",
                "input": [
                    "layer2.1.relu2"
                ],
                "output": [
                    "layer2.2.conv1"
                ],
                "param": {
                    "input_size": 32,
                    "output_size": 32,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "bias": False
                }
            },  # conv1，输入layer2.1.relu2
            {
                "name": "layer2.2.bn1",
                "type": "BatchNorm2D",
                "input": [
                    "layer2.2.conv1"
                ],
                "output": [
                    "layer2.2.bn1"
                ],
                "param": {
                    "input_size": 32,
                    "output_size": 32
                }
            },  # bn1
            {
                "name": "layer2.2.relu1",
                "type": "ReLU",
                "input": [
                    "layer2.2.bn1"
                ],
                "output": [
                    "layer2.2.relu1"
                ]
            },  # relu1
            {
                "name": "layer2.2.conv2",
                "type": "Conv2D",
                "input": [
                    "layer2.2.relu1"
                ],
                "output": [
                    "layer2.2.conv2"
                ],
                "param": {
                    "input_size": 32,
                    "output_size": 32,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "bias": False
                }
            },  # conv2
            {
                "name": "layer2.2.bn2",
                "type": "BatchNorm2D",
                "input": [
                    "layer2.2.conv2"
                ],
                "output": [
                    "layer2.2.bn2"
                ],
                "param": {
                    "input_size": 32,
                    "output_size": 32
                }
            },  # bn2
            {
                "name": "layer2.2.residual",
                "type": "Residual",
                "input": [
                    "layer2.2.bn2",
                    "layer2.1.relu2"
                ],
                "output": [
                    "layer2.2.residual"
                ]
            },  # residual
            {
                "name": "layer2.2.relu2",
                "type": "ReLU",
                "input": [
                    "layer2.2.residual"
                ],
                "output": [
                    "layer2.2.relu2"
                ]
            },  # relu2,


           # maxpool 'bn2', 'fc'
            {
                "name": "pooling2d",
                "type": "Pooling2D",
                "input": [
                    "layer2.2.relu2"
                ],
                "output": [
                    "pooling2d"
                ],
                "param": {
                    "input_size": 32,
                    "output_size": 32,
                    "global_pooling":True,
                    "padding": 0
                }
            },
            {
                "name": "bn2",
                "type": "BatchNorm2D",
                "input": [
                    "pooling2d"
                ],
                "output": [
                    "bn2"
                ],
                "param": {
                    "input_size": 32,
                    "output_size": 32
                }
            },  # bn2
            {
                "name": "fc",
                "type": "Linear",
                "input": [
                    "bn2"
                ],
                "output": [
                    "fc"
                ],
                "param": {
                    "input_size": 32,
                    "output_size": 7,
                    "bias": True
                }
            },
        ]
    }
    # print (json.dumps({'a': 'Runoob', 'b': 7}, sort_keys=True, indent=4, separators=(',', ': ')))
    with open(write_path, "w") as file:
        json.dump(json_dict, file, indent=4, separators=(',', ': '))
def WritePooling2dJson(write_path):
    pass


def WriteConv2dJson(write_path):
    pass

def WriteBatchNormJson(write_path):
    pass

def WriteLeakyReluJson(write_path):
    pass


if __name__ == "__main__":
    # WriteResidualJson("json/residual1.json")
    # WriteResnet20("json/demo_resnet20.json")
    WriteResnet14("json/ao_resnet14.json")
    # file = open("json/demo_resnet20.json","r")
    # json_str = json.loads(file.read(),encoding="utf-8")
    # operators = json_str["operators"]
    # print(len(operators))
    # print(operators[23])