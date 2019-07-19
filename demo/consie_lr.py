# -*- coding: utf-8 -*-
# @Time    : 2019/4/12 11:06
# @Author  : ljf
import torch
import math
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch import optim
from torch.utils import data

# 超参数
torch.manual_seed(1)
LR = 0.01
Batch_size = 24
Epoch = 24

# 定义数据集
x = torch.unsqueeze(torch.linspace(-1,1,1000),dim=1)
y = x**2 + 0.1*torch.normal(torch.zeros(x.size()))

# 先转换成torch能识别的Dataset >> 再使用数据加载器加载数据
data_set = data.TensorDataset(x,y)
dataset_loader = data.DataLoader(dataset = data_set,
                                             batch_size = Batch_size,
                                             shuffle = True,
                                             num_workers = 2,)
# 定义神经网络
class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.hidden = torch.nn.Linear(1,20)
        self.predict = torch.nn.Linear(20,1)

    def forward(self, input):
        x = F.relu(self.hidden(input))
        x = self.predict(x)
        return x

def adjust_lr(args=None, optimizer=None, epoch=None, state=None,CosineAnnealingLR=None):
    if args.is_warmup and epoch <= args.warmup_epoch:
        warmup_max_lr = args.lr*args.warmup_rate
        state['lr'] = args.lr+(warmup_max_lr-args.lr)*(epoch/args.warmup_epoch)
    else:
        if args.lr_type == "schedule":
            if epoch in args.schedule:
                state['lr'] *= args.gamma
        else:
            if not args.is_warmup:
                args = args._replace(warmup_epoch=0)
            CosineAnnealingLR.step(epoch-args.warmup_epoch-10)
            state['lr'] = CosineAnnealingLR.get_lr()
            # lr =0 + (lr - 0) * (1 + math.cos(math.pi * epoch / args.T_max)) / 2

    for param_group in optimizer.param_groups:
        param_group['lr'] = state['lr']

model = Net()
loss_func = torch.nn.MSELoss()


opt = optim.SGD(lr=0.01,params=model.parameters())

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from collections import namedtuple
    name_tuple = namedtuple("args",["lr","is_warmup","warmup_rate","warmup_epoch","lr_type","schedule","T_max","gamma"])

    config = name_tuple(lr=0.01,is_warmup=True,warmup_rate=4,warmup_epoch=5,lr_type="cos",schedule=[50,80],T_max=10,gamma=0.1)
    # batch_lr = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20, eta_min=0,
    #                                                       last_epoch= - 1)

    state = {"lr":0.01}
    lr_list = []
    CosineAnnealingLR = None
    for epoch in range(100):

        # if config.is_warmup and epoch == config.warmup_epoch and config.lr_type=="cos" or not config.is_warmup and config.lr_type=="cos":
        CosineAnnealingLR = torch.optim.lr_scheduler.CosineAnnealingLR(opt,eta_min=0.01/4, T_max=config.T_max,
                                                                           last_epoch=-1)
        adjust_lr(args=config, optimizer=opt, epoch=epoch, state=state,CosineAnnealingLR=CosineAnnealingLR)
        for param in opt.param_groups:
            get_lr =param["lr"]
        try:
            lr_list.append(get_lr[0])
        except:
            lr_list.append(get_lr)
    print(lr_list)
    plt.plot(list(range(100)),lr_list,"r-")
    plt.show()
