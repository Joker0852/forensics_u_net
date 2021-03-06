import torch
import torch.nn as nn
import torch.nn.functional as F


# ~~~~~~~~~~ U-Net ~~~~~~~~~~

class U_down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(U_down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            U_double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

# class U_up(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(U_up, self).__init__()
#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.conv = U_double_conv(in_ch, out_ch)

#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         diffX = x2.size()[2] - x1.size()[2]
#         diffY = x2.size()[3] - x1.size()[3]

#         x1 = F.pad(x1, (diffY, 0,
#                         diffX, 0))
#         x = torch.cat([x2, x1], dim=1)

#         x = self.conv(x)
#         return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = U_double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class U_double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(U_double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

# 和上边的上采样有所不同，这里就用了一次卷积，上边那个用了两次卷积
class U_up(nn.Module):
    def __init__(self, in_ch, out_ch, scale_factor=2):
        super(U_up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = U_double_conv(in_ch, out_ch)

    def forward(self, x):
        out = self.up(x)
        out = self.conv(out)
        return out