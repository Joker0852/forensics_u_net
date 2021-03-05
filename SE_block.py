import torch.nn as nn
import torch

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
