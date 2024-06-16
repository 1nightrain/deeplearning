import torch
import torch.nn as nn
import torch.nn.functional as F

"""
体积图像的精确边界分割是图像引导诊断和计算机辅助干预的关键任务，特别是对于临床实践中的边界混乱。然而，由于缺乏边界形状约束，U型网络无法有效解决这一挑战。
此外，现有的细化边界的方法过分强调细长结构，由于网络对微小物体建模的能力有限，导致过拟合现象。在本文中，我们通过包含与相邻区域的相互作用动态来重新概念化边界生成机制。
此外，我们提出了一个称为 PnPNet 的统一网络来模拟混淆边界区域的形状特征。 PnPNet 的核心成分包括推分支和拉分支。具体来说，基于扩散理论，我们从推动分支设计了语义差异模块（SDM）来挤压边界区域。
SDM 内的显式和隐式差异信息显着提高了类间边界的表示能力。
此外，在 K-means 算法的推动下，引入了拉分支的类聚类模块（CCM）来拉伸相交的边界区域。
因此，推分支和拉分支将分别缩小和放大边界不确定性。他们提供了两种对抗力量来促进模型输出更精确的边界划分。
我们对三个具有挑战性的公共数据集和一个内部数据集进行了实验，其中包含模型预测中的三种类型的边界混淆。
实验结果证明了 PnPNet 相对于其他分割网络的优越性，特别是在 HD 和 ASSD 的评估指标上。此外，推拉分支可以作为即插即用模块来增强经典的U形基线模型。
"""

class SDC(nn.Module):
    def __init__(self, in_channels, guidance_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):
        super(SDC, self).__init__()
        self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.conv1 = Conv3dbn(guidance_channels, in_channels, kernel_size=3, padding=1)
        # self.conv1 = Conv3dGN(guidance_channels, in_channels, kernel_size=3, padding=1)
        self.theta = theta
        self.guidance_channels = guidance_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        # initialize
        x_initial = torch.randn(in_channels, 1, kernel_size, kernel_size, kernel_size)
        x_initial = self.kernel_initialize(x_initial)

        self.x_kernel_diff = nn.Parameter(x_initial)
        self.x_kernel_diff[:, :, 0, 0, 0].detach()
        self.x_kernel_diff[:, :, 0, 0, 2].detach()
        self.x_kernel_diff[:, :, 0, 2, 0].detach()
        self.x_kernel_diff[:, :, 2, 0, 0].detach()
        self.x_kernel_diff[:, :, 0, 2, 2].detach()
        self.x_kernel_diff[:, :, 2, 0, 2].detach()
        self.x_kernel_diff[:, :, 2, 2, 0].detach()
        self.x_kernel_diff[:, :, 2, 2, 2].detach()

        guidance_initial = torch.randn(in_channels, 1, kernel_size, kernel_size, kernel_size)
        guidance_initial = self.kernel_initialize(guidance_initial)

        self.guidance_kernel_diff = nn.Parameter(guidance_initial)
        self.guidance_kernel_diff[:, :, 0, 0, 0].detach()
        self.guidance_kernel_diff[:, :, 0, 0, 2].detach()
        self.guidance_kernel_diff[:, :, 0, 2, 0].detach()
        self.guidance_kernel_diff[:, :, 2, 0, 0].detach()
        self.guidance_kernel_diff[:, :, 0, 2, 2].detach()
        self.guidance_kernel_diff[:, :, 2, 0, 2].detach()
        self.guidance_kernel_diff[:, :, 2, 2, 0].detach()
        self.guidance_kernel_diff[:, :, 2, 2, 2].detach()

    def kernel_initialize(self, kernel):
        kernel[:, :, 0, 0, 0] = -1

        kernel[:, :, 0, 0, 2] = 1
        kernel[:, :, 0, 2, 0] = 1
        kernel[:, :, 2, 0, 0] = 1

        kernel[:, :, 0, 2, 2] = -1
        kernel[:, :, 2, 0, 2] = -1
        kernel[:, :, 2, 2, 0] = -1

        kernel[:, :, 2, 2, 2] = 1

        return kernel

    def forward(self, x, guidance):
        guidance_channels = self.guidance_channels
        in_channels = self.in_channels
        kernel_size = self.kernel_size

        guidance = self.conv1(guidance)

        x_diff = F.conv3d(input=x, weight=self.x_kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=1,
                          groups=in_channels)

        guidance_diff = F.conv3d(input=guidance, weight=self.guidance_kernel_diff, bias=self.conv.bias,
                                 stride=self.conv.stride, padding=1, groups=in_channels)
        out = self.conv(x_diff * guidance_diff * guidance_diff)
        return out


class SDM(nn.Module):
    def __init__(self, in_channel=3, guidance_channels=2):
        super(SDM, self).__init__()
        self.sdc1 = SDC(in_channel, guidance_channels)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm3d(in_channel)

    def forward(self, feature, guidance):
        boundary_enhanced = self.sdc1(feature, guidance)
        boundary = self.relu(self.bn(boundary_enhanced))
        boundary_enhanced = boundary + feature

        return boundary_enhanced



class Conv3dbn(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )

        bn = nn.BatchNorm3d(out_channels)

        super(Conv3dbn, self).__init__(conv, bn)


class Conv3dGNReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        gelu = nn.GELU()

        gn = nn.GroupNorm(4, out_channels)

        super(Conv3dGNReLU, self).__init__(conv, gn, gelu)


class Conv3dGN(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )

        gn = nn.GroupNorm(4, out_channels)

        super(Conv3dGN, self).__init__(conv, gn)

if __name__ == '__main__':
    block = SDM(in_channel=3, guidance_channels=3)
    input = torch.rand(32, 3, 64, 32, 32)
    guidance = torch.randn((32, 3, 64, 32, 32))

    output = block(input,guidance)
    print(input.size())
    print(output.size())