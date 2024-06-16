# https://github.com/apple/ml-cvnets

"""
以下是该模块的主要组件和操作：

自注意力计算：使用线性变换(fc_i, fc_k, fc_v和fc_o)将输入映射到不同的子空间，并计算权重(weight_i)来为每个查询分配注意力权重。注意力权重通过对fc_i的输出进行softmax操作得到，然后用于加权fc_k(input)的输出，得到context_score。接下来，通过对context_score进行求和，以获得一个上下文向量(context_vector)，该向量用于加权fc_v(input)的输出。最后，对v进行线性变换(fc_o)以获得最终的输出。

初始化权重：通过init_weights方法来初始化模块中的权重。

前向传播：根据输入执行自注意力计算，返回计算得到的注意力输出。
"""

import numpy as np
import torch
from torch import nn
from torch.nn import init


class MobileViTv2Attention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(MobileViTv2Attention, self).__init__()
        self.fc_i = nn.Linear(d_model, 1)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)

        self.d_model = d_model
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

    def forward(self, input):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :return:
        '''
        i = self.fc_i(input)  # (bs,nq,1)
        weight_i = torch.softmax(i, dim=1)  # bs,nq,1
        context_score = weight_i * self.fc_k(input)  # bs,nq,d_model
        context_vector = torch.sum(context_score, dim=1, keepdim=True)  # bs,1,d_model
        v = self.fc_v(input) * context_vector  # bs,nq,d_model
        out = self.fc_o(v)  # bs,nq,d_model

        return out


if __name__ == '__main__':
    block = MobileViTv2Attention(d_model=256)
    # input = torch.rand(64, 64, 512).cuda()
    input = torch.rand(1, 128, 256, 256)
    output = block(input)
    print(input.size(), output.size())
