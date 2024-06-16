# https://github.com/Andrew-Qibin/VisionPermutator

"""
MLP (Multi-Layer Perceptron) 模块：

MLP 是一个多层感知器（MLP）模块，用于将输入数据进行线性变换和激活函数操作，以学习和提取特征。

构造函数 (__init__) 接受以下参数：

in_features：输入特征的维度。
hidden_features：中间隐藏层的特征维度。
out_features：输出层的特征维度。
act_layer：激活函数，默认为 GELU。
drop：Dropout 概率，默认为 0.1。
MLP 模块包括两个线性层（fc1 和 fc2），一个激活函数（act_layer）和一个 Dropout 层（drop）。

forward 方法接受输入 x，首先将输入经过第一个线性层和激活函数，然后应用 Dropout，最后通过第二个线性层得到输出。

WeightedPermuteMLP 模块：

WeightedPermuteMLP 是一个自注意力模块，它用于对输入张量进行特征变换和加权重组。

构造函数 (__init__) 接受以下参数：

dim：输入特征的维度。
seg_dim：分段维度，默认为 8。
qkv_bias：Q、K 和 V 投影是否包括偏差，默认为 False。
proj_drop：投影层后的 Dropout 概率，默认为 0。
WeightedPermuteMLP 模块首先将输入张量通过三个线性层（mlp_c、mlp_h 和 mlp_w）进行特征变换，分别用于通道、高度和宽度方向。

输入张量被分成多个段，并在通道维度上进行重组，然后经过线性层进行特征变换。

每个变换后的段都会计算一个权重，然后通过加权平均的方式将这些段组合在一起，以获得最终的输出。

最终输出通过投影层和 Dropout 进行后处理。

这两个模块通常用于神经网络的不同部分，用于特征提取和建模。MLP 主要用于局部特征的提取，而 WeightedPermuteMLP 主要用于加权重组特征以增强全局特征表示。
"""

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self,in_features,hidden_features,out_features,act_layer=nn.GELU,drop=0.1):
        super().__init__()
        self.fc1=nn.Linear(in_features,hidden_features)
        self.act=act_layer()
        self.fc2=nn.Linear(hidden_features,out_features)
        self.drop=nn.Dropout(drop)

    def forward(self, x) :
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))

class WeightedPermuteMLP(nn.Module):
    def __init__(self,dim,seg_dim=8, qkv_bias=False, proj_drop=0.):
        super().__init__()
        self.seg_dim=seg_dim

        self.mlp_c=nn.Linear(dim,dim,bias=qkv_bias)
        self.mlp_h=nn.Linear(dim,dim,bias=qkv_bias)
        self.mlp_w=nn.Linear(dim,dim,bias=qkv_bias)

        self.reweighting=MLP(dim,dim//4,dim*3)

        self.proj=nn.Linear(dim,dim)
        self.proj_drop=nn.Dropout(proj_drop)
    
    def forward(self,x) :
        B,H,W,C=x.shape

        c_embed=self.mlp_c(x)

        S=C//self.seg_dim
        h_embed=x.reshape(B,H,W,self.seg_dim,S).permute(0,3,2,1,4).reshape(B,self.seg_dim,W,H*S)
        h_embed=self.mlp_h(h_embed).reshape(B,self.seg_dim,W,H,S).permute(0,3,2,1,4).reshape(B,H,W,C)

        w_embed=x.reshape(B,H,W,self.seg_dim,S).permute(0,3,1,2,4).reshape(B,self.seg_dim,H,W*S)
        w_embed=self.mlp_w(w_embed).reshape(B,self.seg_dim,H,W,S).permute(0,2,3,1,4).reshape(B,H,W,C)

        weight=(c_embed+h_embed+w_embed).permute(0,3,1,2).flatten(2).mean(2)
        weight=self.reweighting(weight).reshape(B,C,3).permute(2,0,1).softmax(0).unsqueeze(2).unsqueeze(2)

        x=c_embed*weight[0]+w_embed*weight[1]+h_embed*weight[2]

        x=self.proj_drop(self.proj(x))

        return x



if __name__ == '__main__':
    input=torch.randn(64,8,8,512)
    seg_dim=8
    vip=WeightedPermuteMLP(512,seg_dim)
    out=vip(input)
    print(out.shape)
    