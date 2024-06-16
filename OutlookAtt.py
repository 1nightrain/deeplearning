# https://github.com/sail-sg/volo

"""
以下是该模块的主要组件和操作：

v_pj：通过线性变换将输入特征映射到新的特征空间，以产生 v。

attn：通过线性变换将输入图像的局部区域映射到注意力得分的空间。这个得分表示局部区域的重要性。

attn_drop：一个用于应用注意力得分的丢弃层，以防止过度拟合。

proj 和 proj_drop：用于最终输出的线性变换和丢弃层。

unflod：一个用于手动卷积的操作，将 v 特征张量按指定的 kernel_size、padding 和 stride 进行展开。

pool：用于在输入图像上执行平均池化，以减小图像尺寸。

在前向传播中，模块首先将输入图像的局部区域映射到 v 特征空间，然后计算注意力得分。注意力得分被应用于 v 特征以获得加权特征表示。最后，通过线性变换和丢弃层来进一步处理特征表示，以产生最终的输出。

这个模块的主要用途是捕获输入图像的局部信息，并根据局部区域的重要性来加权特征表示。这对于各种计算机视觉任务，如图像分类和分割，可能都会有所帮助。
"""

import numpy as np
import torch
from torch import nn
from torch.nn import init
import math
from torch.nn import functional as F


class OutlookAttention(nn.Module):

    def __init__(self, dim, num_heads=1, kernel_size=3, padding=1, stride=1, qkv_bias=False,
                 attn_drop=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.scale = self.head_dim ** (-0.5)

        self.v_pj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn = nn.Linear(dim, kernel_size ** 4 * num_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(attn_drop)

        self.unflod = nn.Unfold(kernel_size, padding, stride)  # 手动卷积
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)

    def forward(self, x):
        B, H, W, C = x.shape

        # 映射到新的特征v
        v = self.v_pj(x).permute(0, 3, 1, 2)  # B,C,H,W
        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)
        v = self.unflod(v).reshape(B, self.num_heads, self.head_dim, self.kernel_size * self.kernel_size,
                                   h * w).permute(0, 1, 4, 3, 2)  # B,num_head,H*W,kxk,head_dim

        # 生成Attention Map
        attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # B,H,W,C
        attn = self.attn(attn).reshape(B, h * w, self.num_heads, self.kernel_size * self.kernel_size \
                                       , self.kernel_size * self.kernel_size).permute(0, 2, 1, 3,
                                                                                      4)  # B，num_head，H*W,kxk,kxk
        attn = self.scale * attn
        attn = attn.softmax(-1)
        attn = self.attn_drop(attn)

        # 获取weighted特征
        out = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, C * self.kernel_size * self.kernel_size,
                                                        h * w)  # B,dimxkxk,H*W
        out = F.fold(out, output_size=(H, W), kernel_size=self.kernel_size,
                     padding=self.padding, stride=self.stride)  # B,C,H,W
        out = self.proj(out.permute(0, 2, 3, 1))  # B,H,W,C
        out = self.proj_drop(out)

        return out


# 输入 B, H, W, C,  输出 B, H, W, C
if __name__ == '__main__':
    block = OutlookAttention(dim=256).cuda()
    # input = torch.rand(1, 64, 64, 512).cuda()
    input = torch.rand(1, 128, 256, 256).cuda()
    output = block(input)
    print(input.size(), output.size())
