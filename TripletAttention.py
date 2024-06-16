# https://github.com/mindspore-courses/External-Attention-MindSpore/blob/main/model/attention/TripletAttention.py

"""
以下是这些模块的主要特点和作用：

BasicConv 模块：

这是一个基本的卷积模块，用于进行卷积操作，包括卷积、批归一化（可选）、ReLU 激活函数（可选）。
可以通过参数来控制是否使用批归一化和ReLU激活函数。
ZPool 模块：

这是一个自定义的池化操作，将输入的特征图进行最大池化和平均池化，然后将它们拼接在一起。
AttentionGate 模块：

这个模块实现了一个注意力门控机制，用于学习特征图的注意力权重。
首先通过 ZPool 操作将输入的特征图进行池化。
然后应用一个卷积层，该卷积层输出一个注意力权重，通过 Sigmoid 激活函数将其归一化。
最后，将输入特征图与注意力权重相乘，以得到加权的特征图。
TripletAttention 模块：

这个模块实现了一种三重注意力机制，用于学习特征图的全局和局部信息。
该模块包括三个 AttentionGate 模块，分别用于通道维度（c）、高度维度（h）和宽度维度（w）的注意力权重学习。
可以通过参数 no_spatial 来控制是否忽略空间维度。
最终，将三个注意力权重加权平均，以得到最终的特征图。
"""

import torch
import torch.nn as nn


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1 / 3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1 / 2 * (x_out11 + x_out21)
        return x_out


if __name__ == '__main__':
    input = torch.randn(50, 512, 7, 7)
    triplet = TripletAttention()
    output = triplet(input)
    print(output.shape)
