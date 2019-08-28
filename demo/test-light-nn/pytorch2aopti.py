# -*- coding: utf-8 -*-
# @Time    : 2019/8/14 16:54
# @Author  : ljf
from array import *
from collections import OrderedDict
import sys
import torch

# used for filtering unused weight tensors in multi-task setting
#   for example, filter_tensor_name = ["hidden2lm.weight", "hidden2lm.bias"]
filter_tensor_name = []
# used for specify the weight tensor of conv1d, because we can NOT figure it
#   out automatically, for example, conv1d_tensor_name = ["conv1.weight"]


def get_op_name(tensor_name):
    dot = tensor_name.rfind(".")
    return tensor_name[:dot]


def get_suffix_name(tensor_name):
    dot = tensor_name.rfind(".")
    return tensor_name[dot + 1:]


def is_bias_tensor(name_from_pytorch):
    if "bias" in name_from_pytorch:
        return True
    else:
        return False


def length_normalize(tensor_name, op_name):
    if len(tensor_name) < 64:
        return tensor_name.ljust(64, '\0'), op_name
    else:
        # more than 32, cut it to be 31 (1 for \0)
        need2cut = len(tensor_name) - 63
        res = op_name[:-need2cut] + tensor_name[len(op_name):]
        print("[WARNING] length >= 32, cut tensor_name & op_name!")
        return res.ljust(64, '\0'), op_name[:-need2cut]


def format_name(name):
    op_name = get_op_name(name)
    return length_normalize(name, op_name)

# if len(sys.argv) != 3:
#   print( "usage: {} pytorch_model_file lnn_weight_file".format(sys.argv[0]))
#   sys.exit()
# model_path = "./sgd1-1-ao_resnet-lr0.5-2019-08-15-11_04_47.502572checkpoint.pth.tar"
# res_path = "./data/ao_resnet20.dat"
# model_path = "./pth/sgd1-1-ao_resnet-lr0.5-2019-08-15-11_04_47.502572checkpoint.pth.tar"
# model_path = "./pth/sgd-depth14-ao_resnet-lr0.5-2019-08-27-09_37_09.978241checkpoint.pth.tar"
model_path = "./pth/conv2d.pth"
res_path = "./data/conv2d.dat"
# model = torch.load(
#     model_path,
#     map_location=lambda storage,
#     loc: storage)["state_dict"]
model = torch.load(model_path, map_location=lambda storage, loc : storage)
print(type(model))
print(sum([value.numel() for value in model.values()]))

if isinstance(model, OrderedDict):
    dict = model
else:
    dict = model.state_dict()

tensor_name = []
format_weight_name = []
tensor = []

bias_name = []
bias_tensor = []

ops = []

# 添加到指定的tensor中，例如lstm_rnn
for k, v in dict.items():
    if k in filter_tensor_name:
        print("filter tensor: {}".format(k))
        continue
    #过滤掉bn层的num_batches_tracked
    elif k.endswith("num_batches_tracked"):
        print("filter tensor: {}".format(k))
        continue
    # 替换掉parallel model中的module
    tensor_name.append(k.replace("module.", ""))
    tensor.append(v)

print( "#tensor: {}".format(len(tensor_name)))
# import time
# time.sleep(1000)
f = open(res_path, 'wb')

# write magic & tensor number
magic_arr = array('H')
magic_arr.append(0x1)
magic_arr.append(len(tensor_name))
magic_arr.tofile(f)
print(magic_arr)
# write tensor name & size & value
# name_arr = array('u')
name_str = ""
size_arr = array('I')
val_arr = array('f')
for i in range(len(tensor)):
    # op_name = get_op_name(tensor_name[i])
    # 格式化tensor_name的目的是为了保持name的长度一致
    # 过长的进行截断，但是要保持op_name一致性(不可截断后同属于一个op的op_name不同)
    # tensor_name字符串长度最大为64
    name, op_name = format_name(tensor_name[i])
    print("tensor_{} (of operator [{}]): \torig_name: {}\tshape: {}".format(
        i, op_name, tensor_name[i], tensor[i].size()))
    if op_name not in ops:
        ops.append(op_name)
    # print(len(name))
    format_weight_name.append(name)
    for j in range(len(name)):
        name_str += name[j]
    tensor_i = tensor[i].numpy().flatten().tolist()
    print(
        "\t(of operator [{}])\tfinal_name: {}\tsize: {}\n".format(
            op_name,
            name,
            len(tensor_i)))
    size_arr.append(len(tensor_i))
    val_arr.fromlist(tensor_i)
#  for j in range(len(tensor_i)):
#    if j != len(tensor_i) - 1:
#      print( "%.6f" % tensor_i[j],)
#    else:
#      print("%.6f" % tensor_i[j])
# print((

print("#operator: {}".format(len(ops)))
print(ops)
temp = [name.replace("\0","") for name in format_weight_name]
for n in temp:
    # if n.startswith("layer1.0"):
    #     print(n)
    # elif n.startswith("layer2.0"):
    if n.startswith("bn2"):

        print(n)
# for i in range(len(ops)):
#   print( ops[i])

# name_arr.tofile(f)
f.write(name_str.encode("ascii", "strict"))
size_arr.tofile(f)
val_arr.tofile(f)
f.close()
