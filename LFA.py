import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import torch.nn.functional as F

"""
随着智慧城市的发展，交通流预测（TFP）越来越受到人们的关注。在过去的几年中，基于神经网络的方法在 TFP 方面表现出了令人印象深刻的性能。
然而，之前的大多数研究未能明确有效地模拟流入和流出之间的关系。因此，这些方法通常无法解释且不准确。
在本文中，我们提出了一种用于 TFP 的可解释的局部流注意（LFA）机制，它具有三个优点。 
(1) LFA 具有流量感知能力。与现有的在通道维度上混合流入和流出的作品不同，我们通过一种新颖的注意力机制明确地利用了流量之间的相关性。 
(2) LFA是可解释的。它是根据交通流的真理制定的，学习到的注意力权重可以很好地解释流量相关性。
(3) LFA高效。 LFA没有像之前的研究那样使用全局空间注意力，而是利用局部模式。注意力查询仅在局部相关区域上执行。这不仅降低了计算成本，还避免了错误关注。
"""

class LFA(nn.Module):
    def __init__(self, hidden_channel):
        super(LFA, self).__init__()
        self.proj_hq = nn.Conv2d(in_channels=hidden_channel, out_channels=hidden_channel, kernel_size=1, stride=1,bias=False)
        self.proj_mk = nn.Conv2d(in_channels=hidden_channel, out_channels=hidden_channel, kernel_size=1, stride=1,bias=False)
        self.proj_mv = nn.Conv2d(in_channels=hidden_channel, out_channels=hidden_channel, kernel_size=1, stride=1,bias=False)

        self.kernel_size=7
        self.pad=3

        self.dis=self.init_distance()

    def init_distance(self):
        dis=torch.zeros(self.kernel_size,self.kernel_size).cuda()
        certer_x=int((self.kernel_size-1)/2)
        certer_y = int((self.kernel_size - 1) / 2)
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                ii=i-certer_x
                jj=j-certer_y
                tmp=(self.kernel_size-1)*(self.kernel_size-1)
                tmp=(ii*ii+jj*jj)/tmp+dis[i,j]
                dis[i,j]=torch.exp(-tmp)
        dis[certer_x,certer_y]=0
        return dis


    def forward(self, H,M):
        b,c, h, w = H.shape
        pad_M=F.pad(M,[self.pad,self.pad,self.pad,self.pad])

        Q_h = self.proj_hq(H)  # b,c,h,w
        K_m = self.proj_mk(pad_M)  # b,c,h+2,w+2
        V_m = self.proj_mv(pad_M)  # b,c,h+2,w+2

        K_m=K_m.unfold(2,self.kernel_size,1).unfold(3,self.kernel_size,1)  # b,c,h,w,k,k
        V_m=V_m.unfold(2,self.kernel_size,1).unfold(3,self.kernel_size,1)  # b,c,h,w,k,k

        Q_h=Q_h.permute(0,2,3,1)  # b,h,w,c
        K_m=K_m.permute(0,2,3,4,5,1)  # b,h,w,k,k,c
        K_m=K_m.contiguous().view(b,h,w,-1,c)  # b,h,w,(k*k),c
        alpha=torch.einsum('bhwik,bhwkj->bhwij',K_m,Q_h.unsqueeze(-1))  # b,h,w,(k*k),1
        dis_alpha=self.dis.view(-1,1)  # (k*k),1
        alpha=alpha*dis_alpha
        alpha = F.softmax(alpha.squeeze(dim=-1), dim=-1)  # b,h,w,(k*k)
        V_m=V_m.permute(0,2,3,4,5,1).contiguous().view(b,h,w,-1,c)  # b,h,w,(k*k),c
        res=torch.einsum('bhwik,bhwkj->bhwij',alpha.unsqueeze(dim=-2),V_m)  # b,h,w,1,c
        res=res.permute(0,4,1,2,3).squeeze(-1)  # b,c,h,w
        return res

if __name__ == '__main__':
    hidden_channel = 64  # 隐藏通道数
    block = LFA(hidden_channel).to(device=0)  # 创建 LFA 实例
    input_H = torch.rand(1, hidden_channel, 32, 32).to(device=0)  # 输入H
    input_M = torch.rand(1, hidden_channel, 32, 32).to(device=0)  # 输入M
    output = block(input_H, input_M)  # 模型前向传播
    print("Input shape (H):", input_H.size())
    print("Input shape (M):", input_M.size())
    print("Output shape:   ", output.size())
