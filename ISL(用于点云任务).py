import torch
from torch import nn

"""
学习区域内上下文和区域间关系是加强点云分析特征表示的两种有效策略。然而，现有方法并未充分强调统一两种点云表示策略。
为此，我们提出了一种名为点关系感知网络（PRA-Net）的新颖框架，它由区域内结构学习（ISL）模块和区域间关系学习（IRL）模块组成。 
ISL模块可以动态地将局部结构信息集成到点特征中，而IRL模块通过可微分区域划分方案和基于代表性点的策略自适应且有效地捕获区域间关系。
在形状分类、关键点估计和零件分割等多个 3D 基准上进行的大量实验验证了 PRA-Net 的有效性和泛化能力。
"""

def knn(x, k):
    """
    :param x: (B,3,N)
    :param k: int
    :return: (B,N,k_hat)
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim = 1, keepdim = True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k = k, dim = -1)[1]  # (batch_size, num_points, k_hat)
    return idx


class DFA(nn.Module):
    def __init__(self, features, M=2, r=1, L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            M: the number of branchs.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(DFA, self).__init__()

        self.M = M
        self.features = features
        d = max(int(self.features / r), L)

        self.fc = nn.Sequential(nn.Conv1d(self.features, d, kernel_size=1),
                                nn.BatchNorm1d(d))

        self.fc_out = nn.Sequential(nn.Conv1d(d, self.features, kernel_size=1),
                                    nn.BatchNorm1d(self.features))

    def forward(self, x):
        """
        :param x: [x1,x2] (B,C,N)
        :return:
        """

        shape = x[0].shape
        if len(shape) > 3:
            assert NotImplemented('Don not support len(shape)>=3.')

        # (B,MC,N)
        fea_U = x[0] + x[1]

        fea_z = self.fc(fea_U)
        # B，C，N
        fea_cat = self.fc_out(fea_z)

        attention_vectors = torch.sigmoid(fea_cat)
        fea_v = attention_vectors * x[0] + (1 - attention_vectors) * x[1]

        return fea_v


def get_graph_feature(x, xyz=None, idx=None, k_hat=20):
    """
    Get graph features by minus the k_hat nearest neighbors' feature.
    :param x: (B,C,N)
        input features
    :param xyz: (B,3,N) or None
        xyz coordinate
    :param idx: (B,N,k_hat)
        kNN graph index
    :param k_hat: (int)
        the neighbor number
    :return: graph feature (B,C,N,k_hat)
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(xyz, k=k_hat)  # (batch_size, num_points, k_hat)

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k_hat, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims)
    feature = feature - x
    feature = feature.permute(0, 3, 1, 2)
    return feature


class ISL(nn.Module):

    def __init__(self, in_channel, out_channel_list, k_hat=20, bias=False, ):
        """
        :param in_channel:
            input feature channel type:int
        :param out_channel_list: int or list of int
            out channel of MLPs
        :param k_hat: int
            k_hat in ISL
        :param bias: bool
            use bias or not
        """
        super(ISL, self).__init__()

        out_channel = out_channel_list[0]

        self.self_feature_learning = nn.Conv1d(in_channel // 2, out_channel, kernel_size=1, bias=bias)
        self.neighbor_feature_learning = nn.Conv2d(in_channel // 2, out_channel, kernel_size=1, bias=bias)
        self.k = k_hat

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        last_layer_list = []

        for i in range(len(out_channel_list) - 1):
            in_channel = out_channel_list[i]
            out_channel = out_channel_list[i + 1]
            last_layer_list.append(nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=bias))
            last_layer_list.append(nn.BatchNorm2d(out_channel))
            last_layer_list.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.last_layers = nn.Sequential(*last_layer_list)

        self.bn = nn.BatchNorm2d(out_channel)

        self.bn2 = nn.BatchNorm1d(out_channel)
        self.bn = nn.BatchNorm2d(out_channel)

        self.DFA_layer = DFA(features=out_channel, M=2, r=1)

    def forward(self, x, idx_):
        """
        :param x: (B,3,N)
            Input point cloud
        :param idx_: (B,N,k_hat)
            kNN graph index
        :return: graph feature: (B,C,N,k_hat)
        """

        x_minus = get_graph_feature(x, idx=idx_, k_hat=self.k)
        # (B,C,N,K)
        a1 = self.neighbor_feature_learning(x_minus)
        # (B,C,N)
        a2 = self.self_feature_learning(x)

        a1 = self.leaky_relu(self.bn(a1))
        # (B,C,N)
        a1 = a1.max(dim=-1, keepdim=False)[0]
        a2 = self.leaky_relu(self.bn2(a2))
        res = self.DFA_layer([a1, a2])

        res = self.last_layers(res)

        return res



if __name__ == '__main__':
    block = ISL(in_channel=6, out_channel_list=[3], k_hat=20, bias=False)
    input = torch.rand((2, 3, 100))
    idx = knn(input, k=20)
    output = block(input, idx)
    print(input.size())
    print(output.size())