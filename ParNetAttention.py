# https://github.com/imankgoyal/NonDeepNetworks

"""
模块包括以下组件：

sse（Squeeze-and-Excitation）模块：

通过自适应平均池化将输入张量池化到大小为 1x1。
然后使用一个具有相同通道数的卷积层，产生一组注意力权重，这些权重通过 Sigmoid 激活函数进行缩放。
这些注意力权重用于对输入特征进行加权，以突出重要的特征。
conv1x1 和 conv3x3 模块：

conv1x1 是一个1x1卷积层，用于捕捉输入的全局信息。
conv3x3 是一个3x3卷积层，用于捕捉局部信息。
两者都后跟批归一化层以稳定训练。
silu 激活函数：

Silu（或Swish）激活函数是一种非线性激活函数，它将输入映射到一个非线性范围内。
在前向传播中，输入张量 x 通过这些组件，最终输出特征张量 y。这个模块旨在提高神经网络的特征表示能力，通过不同尺度的特征融合和注意力加权来捕获全局和局部信息。
"""

import numpy as np
import torch
from torch import nn
from torch.nn import init


from einops import rearrange

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)



class ParNetAttention(nn.Module):

    def __init__(self, channel=512):
        super().__init__()
        self.sse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.Sigmoid()
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.BatchNorm2d(channel)
        )
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel)
        )
        self.silu = nn.SiLU()

    def forward(self, x):
        b, c, _, _ = x.size()
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x3 = self.sse(x) * x
        y = self.silu(x1 + x2 + x3)
        return y


#   输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    # input = torch.randn(3, 512, 7, 7).cuda()
    input = torch.randn(1, 128, 256, 256).cuda()
    pna = ParNetAttention(channel=128).cuda()
    output = pna(input)
    print(output.shape)
