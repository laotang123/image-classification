# -*- coding: utf-8 -*-
# @Time    : 2019/7/17 16:50
# @Author  : ljf

import torch
import io
# Save to file
# x = torch.tensor([0, 1, 2, 3, 4])
# torch.save(x, 'tensor.pth')
# # Save to io.BytesIO buffer
# buffer = io.BytesIO()
# torch.save(x, buffer)
print(torch.load("weight-py32.pth"))