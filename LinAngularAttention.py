# https://www.haoranyou.com/castling-vit/

"""
以下是该模块的主要组件和操作：

qkv：这是一个线性层，将输入特征 x 映射到三个不同的线性变换，分别对应查询 (query)，键 (key)，和值 (value)。这三个变换将输入特征的通道划分成多个头 (heads)。

attn_drop 和 proj_drop：这是用于进行注意力矩阵和输出特征的丢弃操作的 Dropout 层。

kq_matmul、kqv_matmul 和 qk_matmul：这些是自定义的矩阵乘法操作，用于计算注意力矩阵中的各个部分。kq_matmul 用于计算键和查询的点积，kqv_matmul 用于计算键和值的点积，qk_matmul 用于计算查询和键的点积。

dconv：这是一个深度卷积层，用于对值进行深度卷积操作。

在前向传播过程中，该模块首先将输入特征 x 映射为查询、键和值。然后，通过上述矩阵乘法操作，计算注意力矩阵的各个部分。接下来，对查询和键进行标准化处理，并计算值的深度卷积。最后，根据注意力矩阵和深度卷积的结果，计算最终的输出特征。

此模块实现了线性角注意力机制，可用于处理序列或图像数据中的信息交互和特征提取任务。该模块的参数配置如 num_heads、qkv_bias、attn_drop 等可以根据具体任务进行调整。
"""

import torch
import torch.nn as nn
import math


class MatMul(nn.Module):
    def __init__(self):
        super(MatMul, self).__init__()

    def forward(self, x, y):
        return torch.matmul(x, y)

class LinAngularAttention(nn.Module):
    def __init__(
        self,
        in_channels,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        res_kernel_size=9,
        sparse_reg=False,
    ):
        super().__init__()
        assert in_channels % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = in_channels // num_heads
        self.scale = head_dim**-0.5
        self.sparse_reg = sparse_reg

        self.qkv = nn.Linear(in_channels, in_channels * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_channels, in_channels)
        self.proj_drop = nn.Dropout(proj_drop)

        self.kq_matmul = MatMul()
        self.kqv_matmul = MatMul()
        if self.sparse_reg:
            self.qk_matmul = MatMul()
            self.sv_matmul = MatMul()

        self.dconv = nn.Conv2d(
            in_channels=self.num_heads,
            out_channels=self.num_heads,
            kernel_size=(res_kernel_size, 1),
            padding=(res_kernel_size // 2, 0),
            bias=False,
            groups=self.num_heads,
        )

    def forward(self, x):
        N, L, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(N, L, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        if self.sparse_reg:
            attn = self.qk_matmul(q * self.scale, k.transpose(-2, -1))
            attn = attn.softmax(dim=-1)
            mask = attn > 0.02 # note that the threshold could be different; adapt to your codebases.
            sparse = mask * attn

        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)
        dconv_v = self.dconv(v)

        attn = self.kq_matmul(k.transpose(-2, -1), v)

        if self.sparse_reg:
            x = (
                self.sv_matmul(sparse, v)
                + 0.5 * v
                + 1.0 / math.pi * self.kqv_matmul(q, attn)
            )
        else:
            x = 0.5 * v + 1.0 / math.pi * self.kqv_matmul(q, attn)
        x = x / x.norm(dim=-1, keepdim=True)
        x += dconv_v
        x = x.transpose(1, 2).reshape(N, L, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


if __name__ == '__main__':
    block = LinAngularAttention(in_channels=128)
    input = torch.rand(32,784,128)
    output = block(input)
    print(input.size(), output.size())
