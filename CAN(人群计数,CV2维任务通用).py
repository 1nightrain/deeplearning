import warnings
warnings.filterwarnings("ignore")
import torch.nn as nn
import torch
from torch.nn import functional as F
from torchvision import models

"""
在本文中，我们提出了两种基于双路径多尺度融合网络（SFANet）和SegNet的改进神经网络，以实现准确高效的人群计数。
受 SFANet 的启发，第一个模型被命名为 M-SFANet，附加了多孔空间金字塔池（ASPP）和上下文感知模块（CAN）。 
M-SFANet 的编码器通过包含具有不同采样率的并行空洞卷积层的 ASPP 进行了增强，因此能够提取目标对象的多尺度特征并合并更大的上下文。
为了进一步处理整个输入图像的尺度变化，我们利用 CAN 模块对上下文信息的尺度进行自适应编码。该组合产生了在密集和稀疏人群场景中进行计数的有效模型。
基于SFANet解码器结构，M-SFANet的解码器具有双路径，用于密度图和注意力图生成。
"""


class ContextualModule(nn.Module):
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super(ContextualModule, self).__init__()
        self.scales = []
        self.scales = nn.ModuleList([self._make_scale(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * 2, out_features, kernel_size=1)
        self.relu = nn.ReLU()
        self.weight_net = nn.Conv2d(features, features, kernel_size=1)
        self._initialize_weights()

    def __make_weight(self, feature, scale_feature):
        weight_feature = feature - scale_feature
        return F.sigmoid(self.weight_net(weight_feature))

    def _make_scale(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        multi_scales = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.scales]
        weights = [self.__make_weight(feats, scale_feature) for scale_feature in multi_scales]
        overall_features = [(multi_scales[0] * weights[0] + multi_scales[1] * weights[1] + multi_scales[2] * weights[
            2] + multi_scales[3] * weights[3]) / (weights[0] + weights[1] + weights[2] + weights[3])] + [feats]
        bottle = self.bottleneck(torch.cat(overall_features, 1))
        return self.relu(bottle)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)




if __name__ == '__main__':
    block = ContextualModule(features=64, out_features=64)
    input_tensor = torch.rand(1, 64, 128, 128)
    output = block(input_tensor)
    print("Input size:", input_tensor.size())
    print("Output size:", output.size())