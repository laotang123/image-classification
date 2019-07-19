# -*- coding: utf-8 -*-
# @Time    : 2019/4/12 13:36
# @Author  : ljf
import torch
import math
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

# 1.parameters
batch_size = 128
base_lr = 0.01
weight_decay = 1e-4
warmup_lr = 4 * base_lr
torch.manual_seed(18)
# 2.dataset and dataload
train_set = datasets.MNIST(root="./data",download=True,train=True,transform=transforms.ToTensor())
test_set = datasets.MNIST(root="./data",download=False,train=False,transform=transforms.ToTensor())
train_load = DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True)
test_load = DataLoader(dataset=test_set,batch_size=batch_size,shuffle=False)

def mixup_data(x, y, alpha=1.0, use_cuda=False):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    # �����ڵ����ݺʹ��ҵ�x���
    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
# 3.net architecture
class Net(nn.Module):
    def __init__(self,num_classes):
        super(Net, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1,16,stride=2,padding=1,kernel_size=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 64, stride=2, padding=1, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.global_avg = nn.AdaptiveMaxPool2d((1,1))
        self.fc = nn.Linear(64,self.num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.global_avg(out)
        out = out.view(out.size(0),-1)
        out = self.fc(out)

        return out
# 4.train and test
def train(data_load,criterion,optimizer,is_mixup,model):
    """return acc,loss
    """
    model.train()
    if is_mixup:
        total_loss = 0
        accuracy = 0
        for id, (inputs, targets) in enumerate(data_load, 1):
            # compute loss

            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=0.0001)
            inputs, targets_a, targets_b = Variable(inputs), Variable(targets_a), Variable(targets_b)
            optimizer.zero_grad()
            outputs = model(inputs)
            pred = torch.argmax(outputs, dim=1)
            loss_func = mixup_criterion(targets_a, targets_b, lam)
            loss = loss_func(criterion, outputs)

            correct = lam * pred.eq(targets_a.data).cpu().float().sum() +(1 - lam) * pred.eq(targets_b.data).cpu().float().sum()
            accuracy += correct/targets.size(0)
            total_loss += loss
            loss.backward()
            optimizer.step()

            if id % 500 == 0:
                print("训练集: Loss:{},Accuracy:{}".format(total_loss / id, accuracy / id))
    else:
        total_loss = 0
        accuracy = 0
        for id ,(inputs,targets) in enumerate(data_load,1):
            outputs = model(inputs)

            # compute loss
            loss = criterion(outputs,targets)
            pred = torch.argmax(outputs, dim=1)
            correct = pred.eq(targets).cpu().sum().float() / targets.size(0)
            accuracy += correct
            total_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            if id % 500 ==0:
                print("训练集: Loss:{},Accuracy:{}".format(total_loss/id,accuracy/id))
def val(data_load, criterion, is_mixup, model):
    """return acc,loss
    """
    model.eval()
    if is_mixup:
        pass
    else:
        total_loss = 0
        accuracy = 0
        for id, (inputs, targets) in enumerate(data_load,1):
            outputs = model(inputs)

            # compute loss
            loss = criterion(outputs, targets)
            total_loss += loss

            pred = torch.argmax(outputs, dim=1)
            correct = pred.eq(targets).cpu().sum().float() / targets.size(0)
            accuracy += correct

            if id % 100 == 0:
                print("测试集 ：Loss:{},Accuracy:{}".format(total_loss / id, accuracy / id))
# 5.adjust the lr
def adjust_lr(optimizer,epoch):
    lr = warmup_lr
    if epoch <= 5:
        # warm-up training for large minibatch
        lr = base_lr + (warmup_lr - base_lr) * epoch / 5.0
    if epoch in [100,150]:
        lr*=0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def find_lr(optimizer,net,criterion,init_value = 1e-8, final_value=10., beta = 0.98):
    num = len(train_load)-1
    mult = (final_value / init_value) ** (1/num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []
    for data in train_load:
        batch_num += 1
        #As before, get the loss for this mini-batch of inputs/outputs
        inputs,labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        #Compute the smoothed loss
        avg_loss = beta * avg_loss + (1-beta) *loss.item()
        smoothed_loss = avg_loss / (1 - beta**batch_num)
        #Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses
        #Record the best loss
        if smoothed_loss < best_loss or batch_num==1:
            best_loss = smoothed_loss
        #Store the values
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))
        #Do the SGD step
        loss.backward()
        optimizer.step()
        #Update the lr for the next step
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
    plt.plot(log_lrs[10:-5],losses[10:-5])
    plt.show()
    # return log_lrs, losses
# 6.main function
def main():
    model = Net(10)
    parameters = filter(lambda x:x.requires_grad,model.parameters())
    optimizer = optim.SGD(parameters,lr=0.04,momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    # logs,losses = find_lr(optimizer,model,criterion)
    # plt.plot(logs[10:-5],losses[10:-5])
    # plt.show()
    for epoch in range(90):
        # adjust_lr(optimizer,epoch)
        train(train_load,criterion,optimizer=optimizer,is_mixup=False,model=model)
        val(test_load,criterion,False,model)

if __name__ == "__main__":
    # main()
    import argparse
    parser = argparse.ArgumentParser(description="kdjk")
    parser.add_argument("--depth",default=5,type=int)
    args = parser.parse_args()
    args.depth = 9
    print(args.depth)