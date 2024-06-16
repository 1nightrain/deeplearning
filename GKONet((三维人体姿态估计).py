from functools import partial
import torch
import torch.nn as nn
from timm.models.layers import DropPath

"""
作为 3D 人体姿态估计 (HPE) 的关键部分，建立 2D 到 3D 提升映射受到深度模糊性的限制。目前大多数工作普遍缺乏对提升映射中相对深度表达和深度模糊误差表达的定量分析，导致预测效率低、可解释性差。
为此，本文基于针孔成像原理挖掘和利用这些表达式的先验几何知识，解耦2D到3D的提升映射并简化模型训练。
具体来说，本文提出了一种具有两分支变压器架构的面向先验几何知识的姿态估计模型，明确引入高维先验几何特征以提高模型效率和可解释性。
它将空间坐标的回归转化为关节之间空间方向向量的预测，以生成多个可行解，进一步减轻深度模糊性。
此外，本文首次提出了一种基于先验几何关系与相对深度表达式解耦的基于非学习的绝对深度估计算法。
它建立从非根节点到根节点的多个独立深度映射来计算绝对深度候选，无参数、即插即用、可解释。
实验表明，所提出的姿态估计模型以更低的参数和更快的推理速度在 Human 3.6M 和 MPI-INF-3DHP 基准上实现了最先进的性能，并且所提出的绝对深度估计算法实现了与传统方法相似的性能，而无需任何网络参数。
"""

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.dim = dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.num_heads = num_heads

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class GKONet(nn.Module):
    def __init__(self,
                 num_joints=17,
                 in_chans=(2, 5),
                 embed_dim_pose=32,
                 embed_dim_joint=128,
                 depth=4,
                 num_heads=8,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 mlp_drop_pose=0.,
                 attn_drop_pose=0.,
                 mlp_drop_joint=0.,
                 attn_drop_joint=0.,
                 drop_path_rate=0.1,
                 norm_layer=None):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.embed_dim_pose = embed_dim_pose
        self.embed_dim_joint = embed_dim_joint

        # embedding
        self.Pose_embedding = nn.Linear(in_chans[0], embed_dim_pose)
        self.Pose_embedding_position = nn.Parameter(torch.zeros(1, num_joints, embed_dim_pose))
        self.Joint_pose_embedding = nn.Linear(in_chans[1] * num_joints, embed_dim_joint)
        self.Joint_embedding_position = nn.Parameter(torch.zeros(1, num_joints, embed_dim_joint))
        self.pos_drop = nn.Dropout(p=0.)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.Pose_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_pose, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=mlp_drop_pose, attn_drop=attn_drop_pose, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.Joint_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_joint, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=mlp_drop_joint, attn_drop=attn_drop_joint, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.merge_pose2joint = nn.ModuleList([
            nn.Linear(embed_dim_pose, embed_dim_joint)
            for i in range(depth)])

        self.merge_joint2pose = nn.ModuleList([
            nn.Linear(embed_dim_joint, embed_dim_pose)
            for i in range(depth)])

        self.Joint_norm = norm_layer(embed_dim_joint + embed_dim_pose)
        self.head_joint = nn.Sequential(
            nn.Linear(embed_dim_joint + embed_dim_pose, 3 * num_joints),
        )

    def forward(self, pose_2d, joint_vector_2d):
        # initiation
        b, j, _ = pose_2d.shape
        joint_vector_2d = joint_vector_2d.reshape(b, j, -1)

        # embedding
        pose_embedding = self.Pose_embedding(pose_2d)
        pose_embedding += self.Pose_embedding_position
        pose_embedding = self.pos_drop(pose_embedding)

        joint_embedding = self.Joint_pose_embedding(joint_vector_2d)
        joint_embedding += self.Joint_embedding_position
        joint_embedding = self.pos_drop(joint_embedding)

        # feature
        for blk_pose, blk_joint, blk_joint2pose, blk_pose2joint in zip(self.Pose_blocks, self.Joint_blocks,
                                                                       self.merge_joint2pose, self.merge_pose2joint):
            pose_embedding_merge = pose_embedding + blk_joint2pose(joint_embedding)
            joint_embedding_merge = joint_embedding + blk_pose2joint(pose_embedding)
            pose_embedding = blk_pose(pose_embedding_merge)
            joint_embedding = blk_joint(joint_embedding_merge)

        # head
        joint_embedding = torch.cat((joint_embedding, pose_embedding), dim=-1)
        joint_embedding = self.Joint_norm(joint_embedding)
        joint_embedding = self.head_joint(joint_embedding)
        joint_embedding = joint_embedding.view(b, j, j, -1)

        joint_vector_filp = -torch.transpose(joint_embedding, 1, 2)
        joint_vector = (joint_embedding + joint_vector_filp) / 2
        feasible_solution = joint_vector[:, :, :1, :] - joint_vector[:, :, :, :]
        final_pose_3d = torch.mean(feasible_solution, dim=1)

        return -joint_vector[:, 0], joint_embedding, final_pose_3d

    def count_flops(self):
        flops = 0
        # embedding
        flops += self.Pose_embedding.in_features * self.Pose_embedding.out_features * 17
        flops += self.Joint_pose_embedding.in_features * self.Joint_pose_embedding.out_features * 17

        # transformer
        for blk_pose in self.Pose_blocks:
            # qkv
            flops += 17 * blk_pose.dim * 3 * blk_pose.dim
            # attn = (q @ k.tanspose(-2, -1))
            flops += blk_pose.num_heads * 17 * (blk_pose.dim // blk_pose.num_heads) * 17
            # x = (attn @ v)
            flops += blk_pose.num_heads * 17 * 17 * (blk_pose.dim // blk_pose.num_heads)
            # proj
            flops += 17 * blk_pose.dim * blk_pose.dim

            # mlp
            flops += blk_pose.mlp.fc1.in_features * blk_pose.mlp.fc1.out_features * 17
            flops += blk_pose.mlp.fc2.in_features * blk_pose.mlp.fc2.out_features * 17

            # norm
            flops += 17 * blk_pose.dim

        for blk_pose in self.Joint_blocks:
            # qkv
            flops += 17 * blk_pose.dim * 3 * blk_pose.dim
            # attn = (q @ k.tanspose(-2, -1))
            flops += blk_pose.num_heads * 17 * (blk_pose.dim // blk_pose.num_heads) * 17
            # x = (attn @ v)
            flops += blk_pose.num_heads * 17 * 17 * (blk_pose.dim // blk_pose.num_heads)
            # proj
            flops += 17 * blk_pose.dim * blk_pose.dim

            # mlp
            flops += blk_pose.mlp.fc1.in_features * blk_pose.mlp.fc1.out_features * 17
            flops += blk_pose.mlp.fc2.in_features * blk_pose.mlp.fc2.out_features * 17

            # norm
            flops += 17 * blk_pose.dim

        # norm
        flops += 17 * self.embed_dim_pose
        flops += 17 * self.embed_dim_joint

        for blk_joint2pose in self.merge_joint2pose:
            flops += blk_joint2pose.in_features * blk_joint2pose.out_features * 17

        for blk_pose2joint in self.merge_pose2joint:
            flops += blk_pose2joint.in_features * blk_pose2joint.out_features * 17

        # head
        flops += self.head_joint[0].in_features * self.head_joint[0].out_features * 17
        flops += 17 * self.embed_dim_pose
        return flops

if __name__ == '__main__':
    block = GKONet()

    # 创建一个随机输入张量
    # 假设每个关节有2个输入通道，共有17个关节
    pose_2d = torch.rand(1, 17, 2)  # (batch_size, num_joints, in_chans[0])
    joint_vector_2d = torch.rand(1, 17, 85)  # (batch_size, num_joints, in_chans[1])

    # 运行模型
    joint_vector, joint_embedding, final_pose_3d = block(pose_2d, joint_vector_2d)

    # 打印输出张量的尺寸
    print("joint_vector size:", joint_vector.size())
    print("joint_embedding size:", joint_embedding.size())
    print("final_pose_3d size:", final_pose_3d.size())

