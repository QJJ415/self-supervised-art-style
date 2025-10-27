# import torch
# import torch.nn.functional as F
# from torch import nn
#
#
# class AttentionFeatureSelector(nn.Module):
#     def __init__(self, feature_dim, num_heads):
#         super(AttentionFeatureSelector, self).__init__()
#         self.num_heads = num_heads
#         self.query = nn.Linear(feature_dim, feature_dim)
#         self.key = nn.Linear(feature_dim, feature_dim)
#         self.value = nn.Linear(feature_dim, feature_dim)
#
#     def forward(self, features):
#         # features: [batch_size, num_patches, feature_dim]
#         # 计算 query, key, value
#         Q = self.query(features)  # [batch_size, num_patches, feature_dim]
#         K = self.key(features)  # [batch_size, num_patches, feature_dim]
#         V = self.value(features)  # [batch_size, num_patches, feature_dim]
#
#         # 计算注意力得分
#         attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)   #  [batch_size, num_patches, num_patches]
#         attn_weights = F.softmax(attn_scores, dim=-1)  # [batch_size, num_patches, num_patches]
#
#         # 加权组合特征
#         weighted_features = torch.matmul(attn_weights, V)  # [batch_size, num_patches, feature_dim]
#
#         # 选择权重最大的局部特征（可选）
#         top_values, top_indices = torch.topk(attn_weights.mean(dim=1), k=5, dim=-1)  # 获取前5个重要的局部特征
#         selected_features = weighted_features[:, top_indices, :]
#
#         return selected_features

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayerSelector(nn.Module):
    def __init__(self, backbone, num_heads=16, feature_dim=768, top_k=4):
        super(AttentionLayerSelector, self).__init__()
        self.backbone = backbone  # 预训练模型的backbone
        self.num_heads = num_heads
        self.top_k = top_k  # 最终选择重要性前k的层

        # 定义一个线性层来计算每层的 Query, Key 和 Value
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)

        # 缩放因子
        self.scale = (feature_dim // num_heads) ** -0.5

        # 输出投影层，用于组合后的特征
        self.proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        print(x.shape)
        # 存储每层的特征
        features = []
        for layer in self.backbone:
            x = layer(x)
            features.append(x)

        print(features[0].shape)  # torch.Size([1, 64, 112, 112])
        # 将每一层的特征 reshape 成 [batch_size, num_patches, feature_dim]
        feature_tensors =features[0].shape  # [batch_size, num_layers, num_patches, feature_dim]
        print(feature_tensors)
        B, L, N, C = feature_tensors

        # 计算 Q, K, V
        Q = self.query(features[0]).view(B, L, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        K = self.key(features[0]).view(B, L, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        V = self.value(features[0]).view(B, L, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # 计算注意力权重
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, num_heads, L, L]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, num_heads, L, L]

        # 加权组合每层特征
        weighted_features = torch.matmul(attn_weights, V)  # [B, num_heads, L, head_dim]

        # 将多头的输出拼接回原始维度
        weighted_features = weighted_features.permute(0, 2, 1, 3).contiguous().view(B, L, C)

        # 投影组合后的特征
        layer_importance = self.proj(weighted_features)  # [B, L, C]

        # 计算层的重要性得分
        importance_scores = layer_importance.mean(dim=(1, 2))  # [B, L]

        # 选择 top_k 层
        top_scores, top_indices = torch.topk(importance_scores, k=self.top_k, dim=-1)
        print("top", top_indices)

        # 提取 top_k 层的特征
        selected_features = feature_tensors[:, top_indices, :, :]  # [B, top_k, num_patches, feature_dim]
        print(len(selected_features))
        print(selected_features.shape)
        return selected_features.mean(dim=1)  # 输出最终组合特征


# 示例使用
backbone = [nn.Conv2d(3, 64, 3, stride=2, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.ReLU()]
model = AttentionLayerSelector(backbone=backbone, feature_dim=768, top_k=4)
input_tensor = torch.randn(1, 3, 224, 224)
output = model(input_tensor)
print(output.shape)  # 输出选择的层特征的组合结果

