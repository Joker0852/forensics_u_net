import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _triple, _pair, _single

# 把全局池化改成方差的自注意力模块
class SE2Layer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE2Layer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = torch.std(x,dim=[2,3],keepdim=True).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# softpool
class soft_pool(nn.Module):
    def __init__(self,kernel_size=8, stride=1):
        super(soft_pool, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
    def forward(self,x):
        kernel_size = self.kernel_size
        stride = self.stride
        _, c, h, w = x.size()
        kernel_size = _pair(kernel_size)
        # Create per-element exponential value sum : Tensor [b x 1 x h x w]
        e_x = torch.sum(torch.exp(x), dim=1, keepdim=True)
        # Apply mask to input and pool and calculate the exponential sum
        # Tensor: [b x c x h x w] -> [b x c x h' x w']
        return F.avg_pool2d(x.mul(e_x), kernel_size, stride=stride).mul_(sum(kernel_size)).div_(
            F.avg_pool2d(e_x, kernel_size, stride=stride).mul_(sum(kernel_size)))

# 权重初始化
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# tensor to image
def tensor_to_np(tensor):
    img = tensor.mul(255).byte()
    img = img.detach().cpu().numpy().transpose((0, 2, 3, 1))
    return img

def tensor_to_np1(tensor):
    img = tensor.mul(255).byte()
    img = img.detach().cpu().numpy().transpose((0, 2, 3, 1))
    return img

# 我们自己定义的0.5矩阵，该矩阵的目的是为了骗过判别器，以此提高生成器的能力，从而经过对抗，也提高判别器的能力
def half_one_matrix():
    label_half = torch.ones((8, 256, 256)) * 0.5
    # label_half = label_half
    return label_half

def zero_label():
    zero_lab = torch.zeros((8, 2, 256, 256))
    return zero_lab

def one_label():
    label = torch.ones((8,2,256,256))
    return label

# 调整学习率
def adjust_learnig_rate(opt, optimizer, epoch):
    lr = opt.lr * (0.5 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        print('Learning rate sets to {}.'.format(param_group['lr']))


# 锁住网络，截断参数的更新
def model_freeze(*args):
    """freeze the network for forward process"""
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = False


# 解锁网络，继续网络的参数更新
def model_unfreeze(*args):
    """ unfreeze the network for parameter update"""
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = True






