# https://github.com/ZjjConan/SimAM

"""
该模块的目的是增强图像特征之间的关系，以提高模型的表现。

以下是模块的主要组件和功能：

初始化：在初始化过程中，模块接受一个参数 e_lambda，它是一个小的正数（默认为1e-4）。e_lambda 用于避免分母为零的情况，以确保数值稳定性。此外，模块还创建了一个 Sigmoid 激活函数 act。

前向传播：在前向传播中，模块执行以下步骤：

计算输入张量 x 的形状信息，包括批量大小 b、通道数 c、高度 h 和宽度 w。
计算像素点的数量 n，即图像的高度和宽度的乘积减去1（减1是因为在计算方差时要排除一个像素的均值）。
计算每个像素点与均值的差的平方，即 (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)，这样可以得到差的平方矩阵。
计算分母部分，即 (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)，并加上小的正数 e_lambda 以确保分母不为零。
计算 y，通过将差的平方矩阵除以分母部分，然后加上0.5。这个操作应用了 Sigmoid 函数，将结果限制在0到1之间。
最后，将输入张量 x 与 y 经过 Sigmoid 激活后的结果相乘，以产生最终的输出。
SIMAM 模块的关键思想是计算每个像素点的特征值与均值之间的关系，并通过 Sigmoid 激活函数来调整这种关系，从而增强特征之间的互动性。这有助于捕获图像中不同位置之间的关系，有助于提高模型性能。
"""

import torch
import torch.nn as nn
from thop import profile

from einops import rearrange

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class Simam_module(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(Simam_module, self).__init__()
        self.act = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.act(y)


# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    model = Simam_module().cuda()
    # x = torch.randn(1, 3, 64, 64).cuda()
    x = torch.randn(32, 784, 128).cuda()
    x = to_4d(x,h=28,w=28)
    y = model(x)
    y = to_3d(y)
    print(y.shape)
    flops, params = profile(model, (x,))
    print(flops / 1e9)
    print(params)
