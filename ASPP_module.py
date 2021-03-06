import torch.nn as nn
from u_net_parts import U_up

momentum=0.0003
mult=1

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.in_channesl = in_channels # 进入aspp的通道数
        self.out_channels = out_channels # filter的个数

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        # 第一个1x1卷积
        self.aspp1 = nn.Conv2d(in_channesl, out_channels, kernel_size=1, stride=1, bias=False)
        # aspp中的空洞卷积，rate=6，12，18
        self.aspp2 = nn.Conv2d(in_channesl, out_channels, kernel_size=3, stride=1,
                               dilation=int(6*mult), padding=int(6*mult),
                               bias=False)
        self.aspp3 = nn.Conv2d(in_channesl, out_channels, kernel_size=3, stride=1,
                               dilation=int(12*mult), padding=int(12*mult),
                               bias=False)
        self.aspp4 = nn.Conv2d(in_channesl, out_channels, kernel_size=3, stride=1,
                               dilation=int(18*mult), padding=int(18*mult),
                               bias=False)
        # 对最后一个特征图进行全局平均池化，再feed给256个1x1的卷积核，都带BN
        self.aspp5 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.aspp1_bn = nn.BatchNorm2d(out_channels, momentum)
        self.aspp2_bn = nn.BatchNorm2d(out_channels, momentum)
        self.aspp3_bn = nn.BatchNorm2d(out_channels, momentum)
        self.aspp4_bn = nn.BatchNorm2d(out_channels, momentum)
        self.aspp5_bn = nn.BatchNorm2d(out_channels, momentum)
        # 先上采样双线性插值得到想要的维度，再进入下面的conv
        self.conv2 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, stride=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum)

    def forward(self, x):
        x1 = self.aspp1(x)
        x1 = self.aspp1_bn(x1)
        x1 = self.relu(x1)
        x2 = self.aspp2(x)
        x2 = self.aspp2_bn(x2)
        x2 = self.relu(x2)
        x3 = self.aspp3(x)
        x3 = self.aspp3_bn(x3)
        x3 = self.relu(x3)
        x4 = self.aspp4(x)
        x4 = self.aspp4_bn(x4)
        x4 = self.relu(x4)
        x5 = self.global_pooling(x)
        x5 = self.aspp5(x5)
        x5 = self.aspp5_bn(x5)
        x5 = self.relu(x5)
        # 上采样：双线性插值使x得到想要的维度
        x5 = U_up(256,256)
        # 经过aspp之后，concat之后通道数变为了5倍
        x = torch.cat((x1, x2, x3, x4, x5), 1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x