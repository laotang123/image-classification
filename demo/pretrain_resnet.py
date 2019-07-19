# -*- coding: utf-8 -*-
# @Time    : 2019/4/17 16:46
# @Author  : ljf
import torchvision.models as models
from torch import nn
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch import optim

# 加载数据集
transform = transforms.Compose([transforms.Resize([480,480]),transforms.ToTensor()])
train_dataset = datasets.MNIST(root="./data",download=True,train=True,transform=transform)
test_dataset = datasets.MNIST(root="./data",download=False,train=False,transform=transform)

train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True,)
test_loader = DataLoader(test_dataset,batch_size=64,shuffle=True)


resnet18 = models.resnet18(pretrained=True)
resnet18.conv1 = nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False)
resnet18.fc = nn.Linear(512,10)


# 正则化
weight_p,bias_p = [],[]
for name,p in resnet18.named_parameters():
    if "weight" in name:
        weight_p += [p]
    else:
        bias_p += [p]
# 优化器
optimizer = optim.SGD([{"params":weight_p,"weight_decay":1e-5},
                       {"params":bias_p,"weight_decay":0}],
                      lr=0.01,momentum=0.9)
# 损失函数
criterion = nn.CrossEntropyLoss()
# 学习率调整 adjust_lr

num_epoches = 100
batch_size = 64
model = resnet18
# 迭代数据进行训练
for epoch in range(1):
    print('epoch {}'.format(epoch + 1))
    print('*' * 10)
    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(train_loader, 1):
        img, label = data
        # 向前传播
        out = resnet18(img)
        loss = criterion(out, label)
        running_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        accuracy = (pred == label).float().mean()
        running_acc += num_correct.item()
        # 向后传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ========================= Log ======================
        step = epoch * len(train_loader) + i
        # (1) Log the scalar values
        info = {'loss': loss.item(), 'accuracy': accuracy.item()}

        if i % 2 == 0:
            print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
                epoch + 1, num_epoches, running_loss / (batch_size * i),
                running_acc / (batch_size * i)))
    print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
        epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(
            train_dataset))))
    model.eval()
    eval_loss = 0
    eval_acc = 0
    for data in test_loader:
        img, label = data
        out = model(img)
        loss = criterion(out, label)
        eval_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.item()
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_dataset)), eval_acc / (len(test_dataset))))
    print()
