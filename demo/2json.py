# -*- coding: utf-8 -*-
# @Time    : 2019/5/29 15:30
# @Author  : ljf

import json
import torch
import numpy as np

with open("./data/net.json","r") as file:
    data = json.load(file)
print(data)
model = torch.load("./data/linear.pth",map_location=lambda storage,loc:storage)
print(model)
# operators = data["operators"]
# for dict_item in operators:
#     print(dict_item)

# 自动接收输入信息
net_dict = {"operators":[],'meta': {'model_version': 1}}


while True:
    # 接受每一层的参数 name，type，input output，param
    layer_dict = {}
    name = input("input name of current layer：")
    if name == "exit":
        break
    layer_dict["name"] = name

    type = input("input type of current layer：")
    if type == "exit":
        break
    layer_dict["type"] = type

    layer_input = input("input input of current layer：")
    if layer_input == "exit":
        break
    layer_dict["input"] = [layer_input]

    output = input("input output of current layer：")
    if output == "exit":
        break
    layer_dict["output"] = [output]

    param = input("have param in current layer?(y/n)：")
    if param == "exit":
        break
    if param == "n":
        net_dict["operators"].append(layer_dict)
        continue
    else:
        layer_dict["param"] = {}
        input_size = input("input input_size of current layer：")
        layer_dict["param"]["input_size"] = input_size

        output_size = input("input output_size of current layer：")
        layer_dict["param"]["output_size"] = output_size

        bias = input("input bias of current layer：")
        layer_dict["param"]["bias"] = bias
        net_dict["operators"].append(layer_dict)

print(net_dict)
json_str = json.dumps(net_dict,indent=4)
with open("./linear.json","w") as f:
    f.write(json_str)

