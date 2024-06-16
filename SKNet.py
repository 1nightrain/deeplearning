# https://github.com/implus/SKNet

"""
该模块的主要功能是对输入张量进行一系列卷积操作，然后计算不同卷积核的注意力权重，并将它们应用于输入的不同部分以生成最终的输出。以下是该模块的主要组件和步骤：

初始化：在初始化中，模块接受以下参数：

channel：输入通道数。
kernels：用于卷积操作的核大小列表。
reduction：通道减少比例，用于降低通道数。
group：卷积操作的分组数。
L：指定的参数，用于确定最大通道数的值。
在初始化过程中，模块创建了一系列卷积层、线性层和 Softmax 操作，以用于后续的计算。

前向传播：在前向传播过程中，模块执行以下步骤：

针对每个核大小，使用相应的卷积操作对输入进行卷积，并将卷积结果存储在列表 conv_outs 中。
将所有卷积结果叠加起来以生成 U，它代表了输入的融合表示。
对 U 进行平均池化，然后通过线性层将通道数减少到 d。
使用线性层计算不同卷积核的注意力权重，并将它们存储在列表 weights 中。
使用 Softmax 函数将注意力权重归一化。
将注意力权重应用于不同卷积核的特征表示，并对它们进行加权叠加，生成最终的输出张量 V。
最终，模块返回张量 V 作为输出。

这个模块的核心思想是在不同尺度的卷积核上计算注意力权重，以捕获输入的多尺度信息，然后将不同尺度的特征进行加权叠加以生成最终的输出。这可以增强模型对不同尺度物体的感知能力。
"""

import torch
from torch import nn
from collections import OrderedDict


class SKAttention(nn.Module):

    def __init__(self, channel=512, kernels=[1, 3, 5, 7], reduction=16, group=1, L=32):
        super().__init__()
        self.d = max(L, channel // reduction)
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(channel, channel, kernel_size=k, padding=k // 2, groups=group)),
                    ('bn', nn.BatchNorm2d(channel)),
                    ('relu', nn.ReLU())
                ]))
            )
        self.fc = nn.Linear(channel, self.d)
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, channel))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs = []
        ### split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats = torch.stack(conv_outs, 0)  # k,bs,channel,h,w

        ### fuse
        U = sum(conv_outs)  # bs,c,h,w

        ### reduction channel
        S = U.mean(-1).mean(-1)  # bs,c
        Z = self.fc(S)  # bs,d

        ### calculate attention weight
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight.view(bs, c, 1, 1))  # bs,channel
        attention_weughts = torch.stack(weights, 0)  # k,bs,channel,1,1
        attention_weughts = self.softmax(attention_weughts)  # k,bs,channel,1,1

        ### fuse
        V = (attention_weughts * feats).sum(0)
        return V

#  输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    input = torch.randn(50, 512, 7, 7)
    se = SKAttention(channel=512, reduction=8)
    output = se(input)
    print(output.shape)
