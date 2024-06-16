import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


"""
多模态情感分析旨在判断互联网用户在各种社交媒体平台上上传的多模态数据的情感。一方面，现有研究关注文本、音频和视觉等多模态数据的融合机制，而忽视了文本与音频、文本与视觉的相似性以及音频与视觉的异质性，导致情感分析出现偏差。
另一方面，多模态数据带来与情感分析无关的噪声，影响融合效果。在本文中，我们提出了一种称为 PS-Mixer 的极向量和强度向量混合模型，它基于 MLP-Mixer，以实现不同模态数据之间更好的通信，以进行多模态情感分析。
具体来说，我们设计了一个极向量（PV）和一个强度向量（SV）来分别判断情绪的极性和强度。 PV是从文本和视觉特征的交流中获得的，以决定情感是积极的、消极的还是中性的。
SV是从文本和音频特征之间的通信中获得的，以分析0到3范围内的情感强度。
此外，我们设计了一个由多个全连接层和激活函数组成的MLP通信模块（MLP-C） ，以使得不同模态特征在水平和垂直方向上充分交互，是利用MLP进行多模态信息通信的新颖尝试。
"""


class MLP_block(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, input_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.net(x)
        return x


class MLP_Communicator(nn.Module):
    def __init__(self, token, channel, hidden_size, depth=1):
        super(MLP_Communicator, self).__init__()
        self.depth = depth
        self.token_mixer = nn.Sequential(
            Rearrange('b n d -> b d n'),
            MLP_block(input_size=channel, hidden_size=hidden_size),
            Rearrange('b n d -> b d n')
        )
        self.channel_mixer = nn.Sequential(
            MLP_block(input_size=token, hidden_size=hidden_size)
        )

    def forward(self, x):
        for _ in range(self.depth):
            x = x + self.token_mixer(x)
            x = x + self.channel_mixer(x)
        return x


if __name__ == '__main__':
    # 创建模型实例
    block = MLP_Communicator(
        token=32,  # token 的大小
        channel=128,  # 通道的大小
        hidden_size=64,  # 隐藏层的大小
        depth=1  # 深度
    )

    # 准备输入张量
    input_tensor = torch.randn(8, 128, 32) #  32与token对应  128与channel对应

    # 执行前向传播
    output_tensor = block(input_tensor)

    # 打印输入张量和输出张量的形状
    print("Input Tensor Shape:", input_tensor.size())
    print("Output Tensor Shape:", output_tensor.size())
