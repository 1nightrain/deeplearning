# https://paperswithcode.com/paper/s-2-mlpv2-improved-spatial-shift-mlp

"""
SplitAttention：

这是一个分离式注意力（Split Attention）模块，用于增强神经网络的特征表示。
参数包括 channel（通道数）和 k（分离的注意力头数）。
在前向传播中，输入张量 x_all 被重塑为形状 (b, k, h*w, c)，其中 b 是批次大小，k 是头数，h 和 w 是高度和宽度，c 是通道数。
然后，计算注意力的权重，通过 MLP 网络计算 hat_a，然后应用 softmax 函数得到 bar_a。
最后，将 bar_a 与输入张量 x_all 相乘，并对所有头的结果进行求和以获得最终的输出。
S2Attention：

这是一个基于Split Attention的注意力模块，用于处理输入张量。
参数包括 channels（通道数）。
在前向传播中，首先对输入张量进行线性变换，然后将结果分为三部分（x1、x2 和 x3）。
接下来，这三部分被传递给 SplitAttention 模块，以计算注意力权重并增强特征表示。
最后，通过另一个线性变换将注意力增强后的特征表示进行合并并返回。
这些模块可以用于构建神经网络中的不同层，以提高特征表示的性能和泛化能力。
"""

import numpy as np
import torch
from torch import nn
from torch.nn import init


def spatial_shift1(x):
    b, w, h, c = x.size()
    x[:, 1:, :, :c // 4] = x[:, :w - 1, :, :c // 4]
    x[:, :w - 1, :, c // 4:c // 2] = x[:, 1:, :, c // 4:c // 2]
    x[:, :, 1:, c // 2:c * 3 // 4] = x[:, :, :h - 1, c // 2:c * 3 // 4]
    x[:, :, :h - 1, 3 * c // 4:] = x[:, :, 1:, 3 * c // 4:]
    return x


def spatial_shift2(x):
    b, w, h, c = x.size()
    x[:, :, 1:, :c // 4] = x[:, :, :h - 1, :c // 4]
    x[:, :, :h - 1, c // 4:c // 2] = x[:, :, 1:, c // 4:c // 2]
    x[:, 1:, :, c // 2:c * 3 // 4] = x[:, :w - 1, :, c // 2:c * 3 // 4]
    x[:, :w - 1, :, 3 * c // 4:] = x[:, 1:, :, 3 * c // 4:]
    return x


class SplitAttention(nn.Module):
    def __init__(self, channel=32, k=3):
        super().__init__()
        self.channel = channel
        self.k = k
        self.mlp1 = nn.Linear(channel, channel, bias=False)
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(channel, channel * k, bias=False)
        self.softmax = nn.Softmax(1)

    def forward(self, x_all):
        b, k, h, w, c = x_all.shape
        x_all = x_all.reshape(b, k, -1, c)  # bs,k,n,c
        a = torch.sum(torch.sum(x_all, 1), 1)  # bs,c
        hat_a = self.mlp2(self.gelu(self.mlp1(a)))  # bs,kc
        hat_a = hat_a.reshape(b, self.k, c)  # bs,k,c
        bar_a = self.softmax(hat_a)  # bs,k,c
        attention = bar_a.unsqueeze(-2)  # #bs,k,1,c
        out = attention * x_all  # #bs,k,n,c
        out = torch.sum(out, 1).reshape(b, h, w, c)
        return out


class S2Attention(nn.Module):

    def __init__(self, channels=32):
        super().__init__()
        self.mlp1 = nn.Linear(channels, channels * 3)
        self.mlp2 = nn.Linear(channels, channels)
        self.split_attention = SplitAttention()

    def forward(self, x):
        b, c, w, h = x.size()
        x = x.permute(0, 2, 3, 1)
        x = self.mlp1(x)
        x1 = spatial_shift1(x[:, :, :, :c])
        x2 = spatial_shift2(x[:, :, :, c:c * 2])
        x3 = x[:, :, :, c * 2:]
        x_all = torch.stack([x1, x2, x3], 1)
        a = self.split_attention(x_all)
        x = self.mlp2(a)
        x = x.permute(0, 3, 1, 2)
        return x


#   输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    input = torch.randn(64, 32, 7, 7)
    s2att = S2Attention(channels=32)
    output = s2att(input)
    print(output.shape)
