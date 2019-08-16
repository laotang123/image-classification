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
        "operator": [
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
                    "output_size": 3,
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
                    "input_size": 3,
                    "output_size": 3
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
            # layer
            {
                "name": "layer.0.conv1",
                "type": "Conv2D",
                "input": [
                    "leakyrelu"
                ],
                "output": [
                    "layer.0.conv1"
                ],
                "param": {
                    "input_size": 3,
                    "output_size": 6,
                    "kernel_size": 3,
                    "stride": 2,
                    "padding": 1,
                    "bias": False
                }
            },  # conv2，输入leakyrelu
            {
                "name": "layer.0.bn1",
                "type": "BatchNorm2D",
                "input": [
                    "layer.0.conv1"
                ],
                "output": [
                    "layer.0.bn1"
                ],
                "param": {
                    "input_size": 3,
                    "output_size": 3
                }
            },  # bn2
            {
                "name": "layer.0.relu1",
                "type": "ReLU",
                "input": [
                    "layer.0.bn1"
                ],
                "output": [
                    "layer.0.relu"
                ]
            },  # relu1
            {
                "name": "layer.0.conv2",
                "type": "Conv2D",
                "input": [
                    "layer.0.relu1"
                ],
                "output": [
                    "layer.0.conv2"
                ],
                "param": {
                    "input_size": 6,
                    "output_size": 6,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "bias": False
                }
            },  # conv3
            {
                "name": "layer.0.bn2",
                "type": "BatchNorm2D",
                "input": [
                    "layer.0.conv2"
                ],
                "output": [
                    "layer.0.bn2"
                ],
                "param": {
                    "input_size": 6,
                    "output_size": 6
                }
            },  # bn2
            # downsample
            {
                "name": "layer.0.downsample.0",
                "type": "Conv2D",
                "input": [
                    "leakyrelu"
                ],
                "output": [
                    "layer.0.downsample.0"
                ],
                "param": {
                    "input_size": 3,
                    "output_size": 6,
                    "kernel_size": 1,
                    "stride": 2,
                    "padding": 1,
                    "bias": False
                }
            },  # conv3
            {
                "name": "layer.0.downsample.1",
                "type": "BatchNorm2D",
                "input": [
                    "layer.0.downsample.0"
                ],
                "output": [
                    "layer.0.downsample.1"
                ],
                "param": {
                    "input_size": 6,
                    "output_size": 6
                }
            },  # bn3
            {
                "name": "layer.0.relu2",
                "type": "ReLU",
                "input": [
                    "layer.0.bn2",
                    "layer.0.downsample.1"
                ],
                "output": [
                    "output"
                ]
            },  # relu2,输出out1
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
    WriteResidualJson("json/residual.json")
