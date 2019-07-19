import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
def one_hot_embedding(labels, num_classes):
    y = torch.eye(num_classes)  # [D,D]
    return y[labels]  # [N,D]
def focal_loss( x, y):
    '''Focal loss.

    Args:
      x: (tensor) sized [N,D].
      y: (tensor) sized [N,].

    Return:
      (tensor) focal loss.
    '''
    alpha = 0.25
    gamma = 2
    num_classes = 3

    t = one_hot_embedding(y.data.cpu(),  num_classes)  # [N,21]
    #t = t[:, 1:]  # exclude background
    t = Variable(t)  # [N,20]

    p = x.sigmoid()
    pt = p * t + (1 - p) * (1 - t)  # pt = p if t > 0 else 1-p
    w = alpha * t + (1 - alpha) * (1 - t)  # w = alpha if t > 0 else 1-alpha
    w = w * (1 - pt).pow(gamma)
    return F.binary_cross_entropy_with_logits(x, t, w, size_average=False)
# model = focal_loss()
inputs = torch.Tensor(np.random.rand(10,3).astype(np.float32))
targets = torch.LongTensor(np.random.randint(0,3,(10)).astype(np.int32))
loss = focal_loss(inputs,targets)
print(loss)