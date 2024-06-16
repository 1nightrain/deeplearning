# https://github.com/mindspore-courses/External-Attention-MindSpore/blob/main/model/attention/UFOAttention.py

"""
以下是这个模块的主要特点和作用：

多头自注意力：这个模块使用了多头自注意力机制，通过将输入进行不同线性变换，分为多个头来计算注意力。h 参数表示注意力头的数量。

线性变换：模块中的线性层（fc_q、fc_k、fc_v 和 fc_o）用于将输入进行线性变换，以生成查询（Q）、键（K）和值（V）的向量。

权重初始化：模块中的线性层的权重被初始化，以确保良好的训练收敛性。这些初始化方法包括卷积层的 He 初始化和线性层的正态分布初始化。

注意力计算：通过计算 Q 和 K 的点积，然后应用归一化函数，得到注意力矩阵。在这个模块中，注意力矩阵经过了一些自定义的归一化（XNorm 函数）。

多头特征整合：多个注意力头的输出被整合在一起，然后通过线性层进行进一步的处理，以生成最终的输出。

Dropout 正则化：模块中使用了 Dropout 操作，以减少过拟合的风险。

参数化的缩放因子：模块中包括一个可学习的缩放因子 gamma，用于调整注意力计算的缩放。

总的来说，UFOAttention模块提供了一种用于神经网络中的自注意力机制，它可以根据输入数据生成不同的查询、键和值，并计算注意力矩阵，然后整合多个头的输出以生成最终的特征表示。这种模块通常用于处理序列数据，如自然语言处理中的 Transformer 模型中的注意力层。
"""

import numpy as np
import torch
from torch import nn
from torch.functional import norm
from torch.nn import init


def XNorm(x, gamma):
    norm_tensor = torch.norm(x, 2, -1, True)
    return x * gamma / norm_tensor


class UFOAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(UFOAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gamma = nn.Parameter(torch.randn((1, h, 1, 1)))

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        kv = torch.matmul(k, v)  # bs,h,c,c
        kv_norm = XNorm(kv, self.gamma)  # bs,h,c,c
        q_norm = XNorm(q, self.gamma)  # bs,h,n,c
        out = torch.matmul(q_norm, kv_norm).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)

        return out


if __name__ == '__main__':
    block = UFOAttention(d_model=512, d_k=512, d_v=512, h=8).cuda()
    input = torch.rand(64, 64, 512).cuda()
    output = block(input, input, input)
    print(input.size(), output.size())
