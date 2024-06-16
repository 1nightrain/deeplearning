# https://github.com/JierunChen/FasterNet

"""
这个代码实现了一个名为Partial_conv3的自定义卷积模块，它根据参数的不同执行不同的操作。这个模块的主要特点如下：

部分卷积操作：这个模块使用了一个nn.Conv2d的部分卷积操作，其中dim_conv3表示卷积操作的输出通道数，通常是输入通道数dim的一部分。这部分卷积操作在输入图像的特定通道上执行。

前向传播策略：这个模块可以采用两种不同的前向传播策略，具体取决于forward参数的设置：

'slicing'：在前向传播时，仅对输入张量的部分通道进行部分卷积操作。这对应于仅在推理时使用部分卷积。
'split_cat'：在前向传播时，将输入张量分为两部分，其中一部分进行部分卷积操作，然后将两部分重新连接。这对应于在训练和推理过程中都使用部分卷积。
部分卷积操作的应用：部分卷积操作被用于输入张量的部分通道上，而保持其他通道不变。这有助于模型有选择性地应用卷积操作到特定通道上，从而可以灵活地控制特征的提取和传播。

残差连接：在部分卷积操作之后，模块保留了未经处理的部分通道，然后将两部分连接起来，以保持输入和输出的通道数一致，以便与其他模块连接。

总的来说，Partial_conv3模块提供了一种自定义卷积策略，可以根据应用的需要选择性地应用卷积操作到输入图像的特定通道上。这种模块可以用于特征选择、通道交互等任务，增加了神经网络的灵活性。
"""


from torch import nn
import torch

from einops.einops import rearrange

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()  # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x


if __name__ == '__main__':
    block = Partial_conv3(128, 2, 'split_cat')
    input = torch.rand(32, 784, 128)
    input = to_4d(input, 28, 28)
    output = block(input)
    output = to_3d(input)
    print(input.size())
    print(output.size())
