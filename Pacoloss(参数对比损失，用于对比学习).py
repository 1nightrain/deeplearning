import torch
import torch.nn as nn

"""
在本文中，我们提出了参数对比学习 (PaCo) 来解决长尾识别问题。基于理论分析，我们观察到监督对比损失倾向于偏向高频类别，从而增加了不平衡学习的难度。我们引入了一组参数化的类可学习中心，从优化的角度重新平衡。
此外，我们在平衡设置下分析了我们的 PaCo 损失。我们的分析表明，随着更多样本与其相应的中心被拉到一起，PaCo 可以自适应地增强将同一类别的样本推近的强度，并有利于硬示例学习。
在长尾 CIFAR、ImageNet、Places 和 iNaturalist 2018 上的实验展现了长尾识别的最新进展。
"""


class PaCoLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.0, supt=1.0, temperature=1.0, base_temperature=None, K=128,
                 num_classes=1000):
        super(PaCoLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = temperature if base_temperature is None else base_temperature
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.supt = supt
        self.num_classes = num_classes

    def forward(self, features, labels=None, sup_logits=None):
        device = torch.device('cuda' if features.is_cuda else 'cpu')

        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels[:batch_size], labels.T).float().to(device)

        # compute logits using complete features tensor
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)

        # add supervised logits
        anchor_dot_contrast = torch.cat(((sup_logits) / self.supt, anchor_dot_contrast), dim=1)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask

        # add ground truth
        one_hot_label = torch.nn.functional.one_hot(labels[:batch_size, ].view(-1, ), num_classes=self.num_classes).to(
            torch.float32)
        mask = torch.cat((one_hot_label * self.beta, mask * self.alpha), dim=1)

        # compute log_prob
        logits_mask = torch.cat((torch.ones(batch_size, self.num_classes).to(device), self.gamma * logits_mask), dim=1)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss


if __name__ == '__main__':
    # 初始化 PaCoLoss 模型
    block = PaCoLoss()

    # 随机生成输入特征、标签和监督logits
    input_features = torch.rand(64, 64)  # 例如，64 个样本的特征
    labels = torch.randint(0, 10, (64,))  # 例如，64 个样本的标签
    sup_logits = torch.rand(64, 1000)  # 例如，64 个样本的监督 logits

    print("Supervised logits shape:", sup_logits.size())

    # 使用输入数据计算损失
    loss = block(input_features, labels=labels, sup_logits=sup_logits)

    # 输出输入特征的形状和计算得到的损失值
    print("Input features shape:", input_features.size())
    print("Output loss:", loss.item())

