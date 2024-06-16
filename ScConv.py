# https://github.com/cheng-haha/ScConv

"""
GroupBatchnorm2d：

这是一个自定义的批量归一化（Batch Normalization）模块。
它支持将通道分组，即将通道分成多个组，每个组共享统计信息。
参数包括 c_num（通道数），group_num（分组数），和 eps（防止除以零的小值）。
在前向传播中，它首先将输入张量按组进行划分，并在每个组内计算均值和标准差，然后使用这些统计信息来对输入进行标准化。
SRU（Self-Reconstruction Unit）：

这是一个自定义的模块，用于增强神经网络的特征表示。
参数包括 oup_channels（输出通道数），group_num（分组数），gate_treshold（门控阈值），和 torch_gn（是否使用PyTorch的GroupNorm）。
在前向传播中，它首先应用分组归一化（Group Normalization），然后通过门控机制（Gate）重新构造输入特征。
门控机制根据输入特征的分布和权重来决定哪些信息被保留，哪些信息被舍弃。
CRU（Channel Reorganization Unit）：

这是一个自定义的通道重组模块，用于重新组织神经网络的通道。
参数包括 op_channel（输出通道数），alpha（通道划分比例），squeeze_radio（压缩比例），group_size（分组大小），和 group_kernel_size（分组卷积核大小）。
在前向传播中，它首先将输入通道分成两部分，然后对每部分进行压缩（squeeze）操作和分组卷积（Group Convolution）操作，最后将结果进行融合。
ScConv（Scale and Channel Convolution）：

这是一个结合了SRU和CRU的模块，用于增强特征表示并进行通道重组。
参数包括 SRU 和 CRU 模块的参数。
在前向传播中，它首先应用SRU模块，然后应用CRU模块，以改善特征表示并重新组织通道。
这些自定义模块可以用于构建更复杂的神经网络，以满足特定的任务和需求。模块中的操作和机制可以帮助提高神经网络的性能和泛化能力。
"""

import torch
import torch.nn.functional as F
import torch.nn as nn


class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


class SRU(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 torch_gn: bool = False
                 ):
        super().__init__()

        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.weight / torch.sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1)
        reweigts = self.sigomid(gn_x * w_gamma)
        # Gate
        info_mask = reweigts >= self.gate_treshold
        noninfo_mask = reweigts < self.gate_treshold
        x_1 = info_mask * gn_x
        x_2 = noninfo_mask * gn_x
        x = self.reconstruct(x_1, x_2)
        return x

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


class CRU(nn.Module):
    '''
    alpha: 0<alpha<1
    '''

    def __init__(self,
                 op_channel: int,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
        # up
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        # low
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                              bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Split
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        # Transform
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        # Fuse
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2


class ScConv(nn.Module):
    def __init__(self,
                 op_channel: int,
                 group_num: int = 4,
                 gate_treshold: float = 0.5,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.SRU = SRU(op_channel,
                       group_num=group_num,
                       gate_treshold=gate_treshold)
        self.CRU = CRU(op_channel,
                       alpha=alpha,
                       squeeze_radio=squeeze_radio,
                       group_size=group_size,
                       group_kernel_size=group_kernel_size)

    def forward(self, x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x


# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    # x = torch.randn(1, 32, 16, 16)
    x = torch.randn(1, 128, 256, 256)
    model = ScConv(128)
    x = model(x)
    # x = torch.unsqueeze(x[:, 0], 1)
    # print(type(x))
    print(x.shape)
