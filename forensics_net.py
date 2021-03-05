import torch.nn as nn
from u_net_parts import U_down,U_up,U_double_conv,inconv
from deform_conv import DeformConv2D
from SE_block import SE2Layer
from ASPP_module import ASPP

class forensics_net(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(Unet, self).__init__()
        # 下采样
        self.inc = inconv(n_channels, 16)
        self.down1 = U_down(16, 32)
        self.offset1 = U_double_conv(32,18)
        self.down2 = DeformConv2D(32, 64)
        self.offset2 = U_double_conv(64,18)
        self.down3 = DeformConv2D(64, 128)
        self.offset3 = U_double_conv(128,18)
        self.down4 = DeformConv2D(128, 256)
        # 上采样
        self.up1 = U_up(256, 128)
        self.up2 = U_up(128, 64)
        self.up3 = U_up(64, 32)
        self.up4 = U_up(32, 16)
        self.out = outconv(16, n_classes)
        # 自注意力
        self.se1 = SE2Layer(16)
        self.se2 = SE2Layer(32)
        self.se3 = SE2Layer(64)
        self.se4 = SE2Layer(128)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)
        return x