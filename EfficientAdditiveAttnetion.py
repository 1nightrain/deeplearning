# https:// tinyurl.com/ 5ft8v46w
"""
以下是这个模块的主要特点和作用：

线性变换：模块中包括两个线性层 to_query 和 to_key，分别用于将输入的特征进行线性变换，将特征维度从 in_dims 映射到 token_dim * num_heads。这两个线性层的输出用于计算查询（Query）和键（Key）。

可学习的权重：模块中包括一个可学习的权重向量 w_g，用于计算加性注意力的权重。这个权重向量的形状是 (token_dim * num_heads, 1)。

归一化：通过 torch.nn.functional.normalize 函数对查询（Query）和键（Key）进行 L2 归一化，以确保它们具有单位长度。

权重计算：计算查询（Query）与权重向量 w_g 的点积，并乘以缩放因子 scale_factor（通常是 token_dim 的倒数的平方根），以得到加性注意力的权重 A。

归一化：对权重 A 进行归一化，以确保它们在序列长度维度上的和为 1。

加权求和：通过将注意力权重 A 与查询（Query）相乘，然后在序列长度维度上求和，得到全局上下文向量 G。

扩展 G：通过 einops.repeat 操作，将全局上下文向量 G 扩展为与键（Key）相同形状的张量。

注意力计算：通过将扩展后的 G 与键（Key）相乘，然后加上原始查询（Query），得到注意力加权的输出。

投影层：通过线性层 Proj 对注意力加权的输出进行投影，将特征维度从 token_dim * num_heads 投影回 token_dim * num_heads。

最终投影：通过线性层 final 对投影后的输出进行最终的线性变换，将特征维度从 token_dim * num_heads 投影回 token_dim，并得到最终的输出。

总的来说，这个模块实现了一种高效的加性注意力机制，用于学习输入序列的全局上下文信息，并将加权后的全局上下文信息与原始特征进行融合，生成最终的输出特征。这种模块通常用于自注意力机制的一部分，可以用于处理序列数据，如自然语言处理中的 Transformer 模型。
"""
import torch
import torch.nn as nn
import einops


class EfficientAdditiveAttnetion(nn.Module):
    """
    Efficient Additive Attention module for SwiftFormer.
    Input: tensor in shape [B, N, D]
    Output: tensor in shape [B, N, D]
    """

    def __init__(self, in_dims=512, token_dim=256, num_heads=2):
        super().__init__()

        self.to_query = nn.Linear(in_dims, token_dim * num_heads)
        self.to_key = nn.Linear(in_dims, token_dim * num_heads)

        self.w_g = nn.Parameter(torch.randn(token_dim * num_heads, 1))
        self.scale_factor = token_dim ** -0.5
        self.Proj = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.final = nn.Linear(token_dim * num_heads, token_dim)

    def forward(self, x):
        query = self.to_query(x)
        key = self.to_key(x)

        query = torch.nn.functional.normalize(query, dim=-1)  # BxNxD
        key = torch.nn.functional.normalize(key, dim=-1)  # BxNxD

        query_weight = query @ self.w_g  # BxNx1 (BxNxD @ Dx1)
        A = query_weight * self.scale_factor  # BxNx1

        A = torch.nn.functional.normalize(A, dim=1)  # BxNx1

        G = torch.sum(A * query, dim=1)  # BxD

        G = einops.repeat(
            G, "b d -> b repeat d", repeat=key.shape[1]
        )  # BxNxD

        out = self.Proj(G * key) + query  # BxNxD

        out = self.final(out)  # BxNxD

        return out


# 输入 B N C ,  输出 B N C
if __name__ == '__main__':
    block = EfficientAdditiveAttnetion(64, 32).cuda()
    input = torch.rand(3, 64 * 64, 64).cuda()
    output = block(input)
    print(input.size(), output.size())
