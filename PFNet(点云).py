import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

"""
在本文中，我们提出了一种点分形网络（PF-Net），这是一种基于学习的新型方法，用于精确和高保真度的点云完成。
与现有的点云完成网络不同，现有的点云完成网络从不完整的点云生成点云的整体形状，并且总是改变现有点并遇到噪声和几何损失，PF-Net保留了不完全点云的空间排列，并可以计算出预测中缺失区域的详细几何结构。
为了成功完成这项任务，PF-Net利用基于特征点的多尺度发电网络，对缺失的点云进行分层估计。此外，我们将多阶段完成损失和对抗性损失相加，以生成更真实的缺失区域。
对抗性损失可以更好地处理预测中的多种模式。我们的实验证明了我们的方法在几个具有挑战性的点云完成任务中的有效性。
"""


class Convlayer(nn.Module):
    def __init__(self, point_scales):
        super(Convlayer, self).__init__()
        self.point_scales = point_scales
        self.conv1 = torch.nn.Conv2d(1, 64, (1, 3))
        self.conv2 = torch.nn.Conv2d(64, 64, 1)
        self.conv3 = torch.nn.Conv2d(64, 128, 1)
        self.conv4 = torch.nn.Conv2d(128, 256, 1)
        self.conv5 = torch.nn.Conv2d(256, 512, 1)
        self.conv6 = torch.nn.Conv2d(512, 1024, 1)
        self.maxpool = torch.nn.MaxPool2d((self.point_scales, 1), 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn6 = nn.BatchNorm2d(1024)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x_128 = F.relu(self.bn3(self.conv3(x)))
        x_256 = F.relu(self.bn4(self.conv4(x_128)))
        x_512 = F.relu(self.bn5(self.conv5(x_256)))
        x_1024 = F.relu(self.bn6(self.conv6(x_512)))
        x_128 = torch.squeeze(self.maxpool(x_128), 2)
        x_256 = torch.squeeze(self.maxpool(x_256), 2)
        x_512 = torch.squeeze(self.maxpool(x_512), 2)
        x_1024 = torch.squeeze(self.maxpool(x_1024), 2)
        L = [x_1024, x_512, x_256, x_128]
        x = torch.cat(L, 1)
        return x


class Latentfeature(nn.Module):
    def __init__(self, num_scales, each_scales_size, point_scales_list):
        super(Latentfeature, self).__init__()
        self.num_scales = num_scales
        self.each_scales_size = each_scales_size
        self.point_scales_list = point_scales_list
        self.Convlayers1 = nn.ModuleList(
            [Convlayer(point_scales=self.point_scales_list[0]) for i in range(self.each_scales_size)])
        self.Convlayers2 = nn.ModuleList(
            [Convlayer(point_scales=self.point_scales_list[1]) for i in range(self.each_scales_size)])
        self.Convlayers3 = nn.ModuleList(
            [Convlayer(point_scales=self.point_scales_list[2]) for i in range(self.each_scales_size)])
        self.conv1 = torch.nn.Conv1d(3, 1, 1)
        self.bn1 = nn.BatchNorm1d(1)

    def forward(self, x):
        outs = []
        for i in range(self.each_scales_size):
            outs.append(self.Convlayers1[i](x[0]))
        for j in range(self.each_scales_size):
            outs.append(self.Convlayers2[j](x[1]))
        for k in range(self.each_scales_size):
            outs.append(self.Convlayers3[k](x[2]))
        latentfeature = torch.cat(outs, 2)
        latentfeature = latentfeature.transpose(1, 2)
        latentfeature = F.relu(self.bn1(self.conv1(latentfeature)))
        latentfeature = torch.squeeze(latentfeature, 1)
        return latentfeature


class PointcloudCls(nn.Module):
    def __init__(self, num_scales, each_scales_size, point_scales_list, k=40):
        super(PointcloudCls, self).__init__()
        self.latentfeature = Latentfeature(num_scales, each_scales_size, point_scales_list)
        self.fc1 = nn.Linear(1920, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.latentfeature(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = F.relu(self.bn3(self.dropout(self.fc3(x))))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


class _netG(nn.Module):
    def __init__(self, num_scales, each_scales_size, point_scales_list, point_num):
        super(_netG, self).__init__()
        self.point_num = point_num  # 保存输入的点数
        self.latentfeature = Latentfeature(num_scales, each_scales_size, point_scales_list)
        self.fc1 = nn.Linear(1920, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc_final = nn.Linear(256, point_num * 3)  # 最后一个全连接层输出维度为 point_num * 3

    def forward(self, x):
        x = self.latentfeature(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc_final(x)  # 输出维度为 [batch_size, point_num * 3]
        x = x.reshape(-1, self.point_num, 3)  # 重塑为 [batch_size, point_num, 3]
        return x



class _netlocalD(nn.Module):
    def __init__(self, crop_point_num):
        super(_netlocalD, self).__init__()
        self.crop_point_num = crop_point_num
        self.conv1 = torch.nn.Conv2d(1, 64, (1, 3))
        self.conv2 = torch.nn.Conv2d(64, 64, 1)
        self.conv3 = torch.nn.Conv2d(64, 128, 1)
        self.conv4 = torch.nn.Conv2d(128, 256, 1)
        self.maxpool = torch.nn.MaxPool2d((self.crop_point_num, 1), 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(448, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 16)
        self.fc4 = nn.Linear(16, 1)
        self.bn_1 = nn.BatchNorm1d(256)
        self.bn_2 = nn.BatchNorm1d(128)
        self.bn_3 = nn.BatchNorm1d(16)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x_64 = F.relu(self.bn2(self.conv2(x)))
        x_128 = F.relu(self.bn3(self.conv3(x_64)))
        x_256 = F.relu(self.bn4(self.conv4(x_128)))
        x_64 = torch.squeeze(self.maxpool(x_64))
        x_128 = torch.squeeze(self.maxpool(x_128))
        x_256 = torch.squeeze(self.maxpool(x_256))
        Layers = [x_256, x_128, x_64]
        x = torch.cat(Layers, 1)
        x = F.relu(self.bn_1(self.fc1(x)))
        x = F.relu(self.bn_2(self.fc2(x)))
        x = F.relu(self.bn_3(self.fc3(x)))
        x = self.fc4(x)
        return x



if __name__ == '__main__':
    # 假设你有三个不同尺度的输入，尺度分别是2048, 512, 256
    input1 = torch.randn(64, 2048, 3)  # 第一个尺度的输入
    input2 = torch.randn(64, 512, 3)  # 第二个尺度的输入
    input3 = torch.randn(64, 256, 3)  # 第三个尺度的输入

    # 将这三个输入作为一个列表传递给模型
    input_ = [input1, input2, input3]

    # 初始化模型
    netG = _netG(num_scales=3, each_scales_size=1, point_scales_list=[2048, 512, 256], point_num=2048)

    # 调用模型并获取输出
    output = netG(input_)
    print(output.shape)  # 输出的形状应该与 input1 相同


