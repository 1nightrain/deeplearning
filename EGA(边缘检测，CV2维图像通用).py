import warnings
warnings.filterwarnings("ignore")
import torch
if torch.cuda.is_available():
    torch.cuda.init()
import torch.nn.functional as F
import torch.nn as nn


""" Edge-Guided Attention module (EGA)
这个模块实现的是一个基于边缘引导的注意力模块 (Edge-Guided Attention Module, EGA)，主要用于计算机视觉中的图像处理任务，特别是关注于边缘信息的场景。下面是对代码各个部分及其用途的详细解释：

高斯模糊和金字塔
gauss_kernel(channels=3, cuda=True): 创建一个用于高斯模糊的卷积核。
downsample(x): 实现下采样，将图像尺寸减半。
conv_gauss(img, kernel): 对图像进行高斯卷积模糊处理。
upsample(x, channels): 实现上采样，将图像尺寸加倍。
make_laplace(img, channels): 生成拉普拉斯金字塔的一层，捕获高频信息。
make_laplace_pyramid(img, level, channels): 生成整个拉普拉斯金字塔，包含多层高频信息。
这些函数的组合用于提取和处理图像的多尺度特征，特别是高频边缘信息，这对于图像分割和检测等任务非常有用。

CBAM (Convolutional Block Attention Module)
ChannelGate: 用于生成通道注意力权重，基于全局平均池化和最大池化。
SpatialGate: 用于生成空间注意力权重，基于最大池化和平均池化的特征融合。
CBAM: 组合ChannelGate和SpatialGate，提供通道和空间的联合注意力机制。
CBAM模块用于增强重要特征，抑制不重要特征，从而提升模型的性能。

EGA (Edge-Guided Attention Module)
EGA类: 主要模块，整合边缘特征、输入特征和预测特征，通过注意力机制融合这些特征，从而增强重要特征的表达。
EGA模块的处理流程如下：

Reverse Attention: 基于预测结果计算背景注意力，并得到背景特征。
Boundary Attention: 通过拉普拉斯金字塔提取预测边缘特征，并生成预测特征。
High-Frequency Feature: 利用高频边缘特征，生成输入特征。
Feature Fusion: 将以上三种特征融合，经过卷积和注意力机制生成融合特征。
Output: 融合特征加上残差，经过CBAM模块，得到最终输出。
主要用途
这个模块主要用于增强图像处理任务中的边缘和高频特征，适用于以下场景：
图像分割: 提高边界精度。
目标检测: 强化边缘特征，提升检测效果。
图像超分辨率: 更好地恢复高频细节。
通过EGA模块，可以更好地捕捉和利用图像中的边缘和细节信息，提升模型在各种图像处理任务中的表现。
"""

def gauss_kernel(channels=3, cuda=True):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    if cuda:
        kernel = kernel.cuda()
    return kernel


def downsample(x):
    return x[:, :, ::2, ::2]


def conv_gauss(img, kernel):
    img = F.pad(img, (2, 2, 2, 2), mode='reflect')
    out = F.conv2d(img, kernel, groups=img.shape[1])
    return out


def upsample(x, channels):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
    cc = cc.permute(0, 1, 3, 2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
    x_up = cc.permute(0, 1, 3, 2)
    return conv_gauss(x_up, 4 * gauss_kernel(channels))


def make_laplace(img, channels):
    filtered = conv_gauss(img, gauss_kernel(channels))
    down = downsample(filtered)
    up = upsample(down, channels)
    if up.shape[2] != img.shape[2] or up.shape[3] != img.shape[3]:
        up = nn.functional.interpolate(up, size=(img.shape[2], img.shape[3]))
    diff = img - up
    return diff


def make_laplace_pyramid(img, level, channels):
    current = img
    pyr = []
    for _ in range(level):
        filtered = conv_gauss(current, gauss_kernel(channels))
        down = downsample(filtered)
        up = upsample(down, channels)
        if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
            up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
        diff = current - up
        pyr.append(diff)
        current = down
    pyr.append(current)
    return pyr


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )

    def forward(self, x):
        avg_out = self.mlp(F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))).unsqueeze(-1).unsqueeze(-1)
        max_out = self.mlp(F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))).unsqueeze(-1).unsqueeze(-1)
        channel_att_sum = avg_out + max_out

        scale = torch.sigmoid(channel_att_sum).expand_as(x)
        return x * scale



class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.spatial = nn.Conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        x_compress = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio)
        self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)
        return x_out


# Edge-Guided Attention Module
class EGA(nn.Module):
    def __init__(self, in_channels):
        super(EGA, self).__init__()

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))

        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid())

        self.cbam = CBAM(in_channels)

    def forward(self, edge_feature, x, pred):
        residual = x
        xsize = x.size()[2:]

        pred = torch.sigmoid(pred)

        # reverse attention
        background_att = 1 - pred
        background_x = x * background_att

        # boudary attention
        edge_pred = make_laplace(pred, 1)
        pred_feature = x * edge_pred

        # high-frequency feature
        edge_input = F.interpolate(edge_feature, size=xsize, mode='bilinear', align_corners=True)
        input_feature = x * edge_input

        fusion_feature = torch.cat([background_x, pred_feature, input_feature], dim=1)
        fusion_feature = self.fusion_conv(fusion_feature)

        attention_map = self.attention(fusion_feature)
        fusion_feature = fusion_feature * attention_map

        out = fusion_feature + residual
        out = self.cbam(out)
        return out

if __name__ == '__main__':
    in_channels = 3
    height, width = 224, 224
    edge_feature = torch.rand(1, in_channels, height, width).cuda()
    x = torch.rand(1, in_channels, height, width).cuda()
    pred = torch.rand(1, 1, height, width).cuda()

    block = EGA(in_channels).cuda()
    output = block(edge_feature, x, pred)

    print("Edge feature size:", edge_feature.size())  # torch.Size([1, 3, 224, 224])
    print("Input feature size:", x.size())            # torch.Size([1, 3, 224, 224])
    print("Prediction size:", pred.size())            # torch.Size([1, 1, 224, 224])
    print("Output size:", output.size())              # torch.Size([1, 3, 224, 224])

