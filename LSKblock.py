# https://github.com/zcablii/Large-Selective-Kernel-Network

"""
以下是该模块的主要组件和操作：

conv0：这是一个深度可分离卷积层，使用 5x5 的卷积核进行卷积操作，groups=dim 意味着将输入的每个通道分为一组进行卷积操作。这一步旨在捕获输入中的空间特征。

conv_spatial：这是另一个深度可分离卷积层，使用 7x7 的卷积核进行卷积操作，stride=1 表示步幅为 1，padding=9 用于零填充操作，groups=dim 表示将输入的每个通道分为一组进行卷积操作，并且通过 dilation=3 进行扩张卷积。这一步旨在捕获输入中的更大范围的空间特征。

conv1 和 conv2：这是两个 1x1 的卷积层，用于降低通道数，将输入的通道数减少到 dim // 2。这两个卷积层分别应用于 conv0 和 conv_spatial 的输出。

conv_squeeze：这是一个 7x7 的卷积层，用于进行通道维度的压缩，将输入通道的数量从 2 降低到 2，通过 sigmoid 函数将输出的值缩放到 (0, 1) 范围内。

conv：这是一个 1x1 的卷积层，用于将通道数从 dim // 2 恢复到 dim，最终的输出通道数与输入的通道数相同。

在前向传播过程中，该模块通过一系列卷积操作将输入的特征图进行加权，其中使用了 sigmoid 权重来调整不同部分的注意力。最终输出的特征图是输入特征图乘以注意力加权的结果。

这个 LSKblock 模块的目的是引入空间和通道注意力，以更好地捕获输入特征图中的重要信息。
"""

import torch
import torch.nn as nn


class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn


# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    block = LSKblock(64).cuda()
    input = torch.rand(1, 64, 64, 64).cuda()
    output = block(input)
    print(input.size(), output.size())
