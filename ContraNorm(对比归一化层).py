import torch
import torch.nn as nn

"""ContraNorm：对比学习视角下的过度平滑及其超越
过度平滑是各种图神经网络 (GNN) 和 Transformer 中的常见现象，其性能会随着层数的增加而下降。我们不是从表示收敛到单个点的完全崩溃的角度来描述过度平滑，而是深入研究维度崩溃的更一般视角，其中表示位于一个狭窄的锥体中。
因此，受到对比学习在防止维度崩溃方面的有效性的启发，我们提出了一种称为 ContraNorm 的新型规范化层。直观地说，ContraNorm 隐式地破坏了嵌入空间中的表示，从而导致更均匀的分布和更轻微的维度崩溃。
在理论分析中，我们证明了 ContraNorm 在某些条件下可以缓解完全崩溃和维度崩溃。我们提出的规范化层可以轻松集成到 GNN 和 Transformer 中，并且参数开销可以忽略不计。
在各种真实数据集上的实验证明了我们提出的 ContraNorm 的有效性。
"""

class ContraNorm(nn.Module):
    def __init__(self, dim, scale=0.1, dual_norm=False, pre_norm=False, temp=1.0, learnable=False, positive=False, identity=False):
        super().__init__()
        if learnable and scale > 0:
            import math
            if positive:
                scale_init = math.log(scale)
            else:
                scale_init = scale
            self.scale_param = nn.Parameter(torch.empty(dim).fill_(scale_init))
        self.dual_norm = dual_norm
        self.scale = scale
        self.pre_norm = pre_norm
        self.temp = temp
        self.learnable = learnable
        self.positive = positive
        self.identity = identity

        self.layernorm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        if self.scale > 0.0:
            xn = nn.functional.normalize(x, dim=2)
            if self.pre_norm:
                x = xn
            sim = torch.bmm(xn, xn.transpose(1,2)) / self.temp
            if self.dual_norm:
                sim = nn.functional.softmax(sim, dim=2) + nn.functional.softmax(sim, dim=1)
            else:
                sim = nn.functional.softmax(sim, dim=2)
            x_neg = torch.bmm(sim, x)
            if not self.learnable:
                if self.identity:
                    x = (1+self.scale) * x - self.scale * x_neg
                else:
                    x = x - self.scale * x_neg
            else:
                scale = torch.exp(self.scale_param) if self.positive else self.scale_param
                scale = scale.view(1, 1, -1)
                if self.identity:
                    x = scale * x - scale * x_neg
                else:
                    x = x - scale * x_neg
        x = self.layernorm(x)
        return x


if __name__ == '__main__':
    block = ContraNorm(dim=128, scale=0.1, dual_norm=False, pre_norm=False, temp=1.0, learnable=False, positive=False, identity=False)
    input = torch.rand(32, 784, 128)
    output = block(input)
    print("Input size:", input.size())
    print("Output size:", output.size())
