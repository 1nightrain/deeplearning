from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath


"""
预测周围车辆的运动对于帮助自动驾驶系统规划安全路径并避免碰撞至关重要。
尽管最近基于LSTM模型通过考虑彼此靠近的车辆之间的运动交互而取得了显着的性能提升，但由于实际复杂驾驶场景中的动态和高阶交互，车辆轨迹预测仍然是一个具有挑战性的研究问题。
为此，我们提出了一种受波叠加启发的社交池（简称波池）方法，用于动态聚合来自本地和全局邻居车辆的高阶交互。
通过将每个车辆建模为具有振幅和相位的波，波池可以更有效地表示车辆的动态运动状态，并通过波叠加捕获它们的高阶动态相互作用。
通过集成Wave-pooling，还提出了一种名为WSiP的基于编码器-解码器的学习框架。
在两个公共高速公路数据集 NGSIM 和 highD 上进行的大量实验通过与当前最先进的基线进行比较来验证 WSiP 的有效性。
更重要的是，WSiP的结果更具可解释性，因为车辆之间的相互作用强度可以通过它们的相位差直观地反映出来。
"""



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PATM(nn.Module):
    def __init__(self, dim, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0. ,mode='fc'):
        super().__init__()

        self.fc_h = nn.Conv2d(dim, dim, 1, 1 ,bias=qkv_bias)
        self.fc_w = nn.Conv2d(dim, dim, 1, 1 ,bias=qkv_bias)
        self.fc_c = nn.Conv2d(dim, dim, 1, 1 ,bias=qkv_bias)

        self.tfc_h = nn.Conv2d( 2 *dim, dim, (1 ,7), stride=1, padding=(0 , 7//2), groups=dim, bias=False)
        self.tfc_w = nn.Conv2d( 2 *dim, dim, (7 ,1), stride=1, padding=( 7//2 ,0), groups=dim, bias=False)
        self.reweight = Mlp(dim, dim // 4, dim * 3)
        self.proj = nn.Conv2d(dim, dim, 1, 1 ,bias=True)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mode =mode
        # 对h和w都学出相位
        if mode=='fc':
            self.theta_h_conv =nn.Sequential(nn.Conv2d(dim, dim, 1, 1 ,bias=True) ,nn.BatchNorm2d(dim) ,nn.ReLU())
            self.theta_w_conv =nn.Sequential(nn.Conv2d(dim, dim, 1, 1 ,bias=True) ,nn.BatchNorm2d(dim) ,nn.ReLU())
        else:
            self.theta_h_conv =nn.Sequential(nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim, bias=False)
                                              ,nn.BatchNorm2d(dim) ,nn.ReLU())
            self.theta_w_conv =nn.Sequential(nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim, bias=False)
                                              ,nn.BatchNorm2d(dim) ,nn.ReLU())



    def forward(self, x):
        B, C, H, W = x.shape
        # C, H, W = x.shape
        # 相位
        theta_h =self.theta_h_conv(x)
        theta_w =self.theta_w_conv(x)
        # Channel-FC提取振幅
        x_h =self.fc_h(x)
        x_w =self.fc_w(x)
        # 用欧拉公式对特征进行展开
        x_h =torch.cat([x_h *torch.cos(theta_h) ,x_h *torch.sin(theta_h)] ,dim=1)
        x_w =torch.cat([x_w *torch.cos(theta_w) ,x_w *torch.sin(theta_w)] ,dim=1)
        # Token-FC
        h = self.tfc_h(x_h)
        w = self.tfc_w(x_w)
        c = self.fc_c(x)
        a = F.adaptive_avg_pool2d(h + w + c ,output_size=1)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)
        x = h * a[0] + w * a[1] + c * a[2]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class WaveBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.BatchNorm2d, mode='fc'):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PATM(dim, qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop ,mode=mode)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


if __name__ == '__main__':
    # 实例化模块并定义输入
    block = WaveBlock(dim=64)  # 假设输入特征的通道数为 64
    input = torch.rand(2, 64, 32, 32)  # 假设输入大小为 (batch_size=2, channels=64, height=32, width=32)

    # 运行前向传播
    output = block(input)

    # 打印输入和输出的大小
    print("输入大小:", input.size())
    print("输出大小:", output.size())