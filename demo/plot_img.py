# -*- coding: utf-8 -*-
# @Time    : 2019/4/17 10:20
# @Author  : ljf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob

def plot(paths,title,save_path):
    print(paths)
    plt.figure()
    plt.title(title)
    color = ["#FF5733","#33FFBD","#33FF57","#FFBD33"]
    for id,path in enumerate(paths):
        result = pd.read_csv(path, sep="\t")
        print(result)
    # # print(result)
    # # print(result.iloc[:,2])
        lr, train_loss, train_acc, = result["Learning Rate"], result["Train Loss"], result["Train Acc"]
        val_loss, val_acc = result["Valid Loss"], result["Valid Acc"]
        x = np.arange(len(lr))
        plt.plot(x, train_acc, color[id],label="lr={}".format(lr[0])+"train_acc")
        plt.plot(x, val_acc, color[id], label="lr={}".format(lr[0]) + "val_acc")
    plt.legend(loc=4)
    plt.savefig(save_path)
    plt.show()
    # plt.suptitle(title,fontsize=12,x = 0.5,y =1)
    #
    # plt.subplot(121)
    # plt.plot(x, val_acc, "c")
    # plt.plot(x, train_acc, "b")
    # plt.legend(loc=2)
    #
    # plt.subplot(122)
    # plt.plot(x, lr, "r")
    # plt.plot(x, train_loss, "g")
    # plt.plot(x, val_loss, "y")
    # plt.legend(loc=1)
    # # plt.show()

paths = glob.glob("./result/classification-8/optimizer/sgd/*.txt")
plot(paths,"resnet20","result-sgd-lr.jpg")


