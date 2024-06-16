# https://github.com/damo-cv/KVT

"""
以下是该模块的主要组件和操作：

qkv：这是一个线性层，将输入特征 x 映射到三个不同的线性变换，分别对应查询 (query)，键 (key)，和值 (value)。这三个变换将输入特征的通道划分成多个头 (heads)。

attn_drop 和 proj_drop：这是用于进行注意力矩阵和输出特征的丢弃操作的 Dropout 层。

topk：这是一个超参数，表示要选择每个查询的前 k 个最相关的键。它控制了 k-最近邻注意力机制的行为。

在前向传播过程中，该模块首先将输入特征 x 映射为查询、键和值。然后，通过矩阵乘法操作计算注意力矩阵，但注意力矩阵的计算在这里进行了修改。具体来说，它使用 torch.topk 函数来选择每个查询的前 k 个最相关的键，然后将其余的注意力权重设为负无穷大，以实现 k-最近邻注意力机制。之后，应用 softmax 归一化得到最终的注意力矩阵。最后，利用注意力矩阵对值进行加权平均，得到最终的输出特征。

这个模块的核心思想是在计算注意力时仅考虑与每个查询最相关的 k 个键，从而减少计算复杂度并提高效率。这对于处理大规模数据或具有长序列的模型特别有用。
"""

import torch
import torch.nn as nn


class kNNAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,topk=100):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.topk = topk

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # the core code block
        mask=torch.zeros(B,self.num_heads,N,N,device=x.device,requires_grad=False)
        index=torch.topk(attn,k=self.topk,dim=-1,largest=True)[1]
        mask.scatter_(-1,index,1.)
        attn=torch.where(mask>0, attn,torch.full_like(attn, float('-inf')))
        # end of the core code block

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


if __name__ == '__main__':
    block = kNNAttention(dim=128)
    input = torch.rand(32,784,128)
    output = block(input)
    print(input.size())
    print(output.size())