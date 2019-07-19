# -*- coding: utf-8 -*-
# @Time    : 2019/5/29 12:06
# @Author  : ljf

import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
parse = argparse.ArgumentParser(description="test socket")

parse.add_argument("--name",default="liu ming",type=str,help="")
parse.add_argument("--age",default=18,type=int,help="")
parse.add_argument("--is_man",default="False",type=str2bool,help="")
person = parse.parse_args()
print("hello world!")
print("name:",person.name)
print("age:",person.age)
print("is_man",person.is_man)