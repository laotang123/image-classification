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
    color = ["#FF5733","#33FFBD","#FFBD33","#33FF57","#000000"]
    legend = ["baseline","C2/4","C3/4"]
    for id,path in enumerate(paths):
        result = pd.read_csv(path, sep="\t")
        print(result)
    # # print(result.iloc[:,2])
        lr, train_loss, train_acc, = result["Learning Rate"], result["Train Loss"], result["Train Acc"]
        val_loss, val_acc = result["Valid Loss"], result["Valid Acc"]
        x = np.arange(len(lr))
        plt.plot(x, train_acc, color[id],label="{}".format(legend[id])+"- train_acc")
        plt.plot(x, val_acc, color[id], linestyle = "-.",label="{}".format(legend[id]) + "- val_acc")
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

if __name__ == "__main__":
    paths = glob.glob("./resultS/*.txt")
    plot(paths, "learner-compare", "meta.jpg")


