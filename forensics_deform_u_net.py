import torch
import torch.nn as nn
from torchvision.models import resnet50
import torch.nn.functional as  F
from deform_conv import DeformConv2D
from utils import SE2Layer

# 这里就是用了一个简单的U-net网络，通过上下采样，再增加一个内容损失来监督，从而去掉风格
class deform_forensics(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(deform_forensics, self).__init__()
        # 下采样
        self.conv1 = DoubleConv(in_channels, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(32, 64)
        self.pool3 = nn.MaxPool2d(2)
        self.offset1 = DoubleConv(64,18)
        self.conv4 = DeformConv2D(64, 128)
        self.pool4 = nn.MaxPool2d(2)
        self.offset2 = DoubleConv(128,18)
        self.conv5 = DeformConv2D(128, 256)
        # 上采样
        self.up_sample1 = better_upsampling(in_ch=256, out_ch=128, scale_factor=2)
        self.up_sample2 = better_upsampling(in_ch=128, out_ch=64, scale_factor=2)
        self.up_sample3 = better_upsampling(in_ch=64, out_ch=32, scale_factor=2)
        self.up_sample4 = better_upsampling(in_ch=32, out_ch=16, scale_factor=2)
        self.conv6 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
        # Se block
        self.se1 = SE2Layer(16)
        self.se2 = SE2Layer(32)
        self.se3 = SE2Layer(64)
        self.se4 = SE2Layer(128)
        

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up_6 = self.better_upsampling(in_ch=256, out_ch=128, scale_factor=2)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = nn.Upsample((c3.shape[2], c3.shape[3]), mode='bilinear',align_corners=True)(up_7)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = nn.Upsample((c2.shape[2], c2.shape[3]), mode='bilinear',align_corners=True)(up_8)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = nn.Upsample((c1.shape[2], c1.shape[3]), mode='bilinear',align_corners=True)(up_9)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        out = self.conv6(c9)
        return out


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class better_upsampling(nn.Module):
    def __init__(self, in_ch, out_ch, scale_factor=2):
        super(better_upsampling, self).__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.up(x)
        out = self.conv(out)
        return out