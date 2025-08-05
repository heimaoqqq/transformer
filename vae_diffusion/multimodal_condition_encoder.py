#!/usr/bin/env python3
"""
多模态条件编码器
结合用户ID、时间信息、频率特征等多种条件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiModalConditionEncoder(nn.Module):
    """多模态条件编码器 - 结合多种条件信息"""
    
    def __init__(self, num_users: int, embed_dim: int = 512, 
                 use_time_encoding: bool = True,
                 use_frequency_encoding: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        self.num_users = num_users
        self.embed_dim = embed_dim
        self.use_time_encoding = use_time_encoding
        self.use_frequency_encoding = use_frequency_encoding
        
        # 1. 用户ID编码
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        
        # 2. 时间编码 (如果启用)
        if use_time_encoding:
            self.time_encoder = nn.Sequential(
                nn.Linear(2, embed_dim // 4),  # sin/cos编码
                nn.GELU(),
                nn.Linear(embed_dim // 4, embed_dim // 2),
            )
        
        # 3. 频率特征编码 (如果启用)
        if use_frequency_encoding:
            self.freq_encoder = nn.Sequential(
                nn.Linear(64, embed_dim // 4),  # 假设64个频率bin
                nn.GELU(),
                nn.Linear(embed_dim // 4, embed_dim // 2),
            )
        
        # 4. 多模态融合
        fusion_input_dim = embed_dim
        if use_time_encoding:
            fusion_input_dim += embed_dim // 2
        if use_frequency_encoding:
            fusion_input_dim += embed_dim // 2
            
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_input_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        
        # 5. 注意力权重学习
        self.attention_weights = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 3),  # 3个模态的权重
            nn.Softmax(dim=-1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.02)
    
    def _get_time_encoding(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """生成时间位置编码"""
        # 随机时间步 (模拟不同时间的数据)
        time_steps = torch.rand(batch_size, 1, device=device) * 2 * np.pi
        
        # Sin/Cos编码
        sin_encoding = torch.sin(time_steps)
        cos_encoding = torch.cos(time_steps)
        time_encoding = torch.cat([sin_encoding, cos_encoding], dim=-1)
        
        return self.time_encoder(time_encoding)
    
    def _get_frequency_encoding(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """生成频率特征编码"""
        # 模拟频率特征 (实际应用中应该从数据中提取)
        freq_features = torch.randn(batch_size, 64, device=device)
        return self.freq_encoder(freq_features)
    
    def forward(self, user_indices: torch.Tensor, 
                time_features: torch.Tensor = None,
                freq_features: torch.Tensor = None) -> torch.Tensor:
        """
        多模态条件编码
        
        Args:
            user_indices: [B] 用户索引
            time_features: [B, 2] 时间特征 (可选)
            freq_features: [B, 64] 频率特征 (可选)
            
        Returns:
            [B, 1, embed_dim] 多模态条件表示
        """
        batch_size = user_indices.shape[0]
        device = user_indices.device
        
        # 1. 用户ID编码
        user_embeds = self.user_embedding(user_indices)  # [B, embed_dim]
        
        modality_features = [user_embeds]
        
        # 2. 时间编码
        if self.use_time_encoding:
            if time_features is not None:
                time_embeds = self.time_encoder(time_features)
            else:
                time_embeds = self._get_time_encoding(batch_size, device)
            modality_features.append(time_embeds)
        
        # 3. 频率编码
        if self.use_frequency_encoding:
            if freq_features is not None:
                freq_embeds = self.freq_encoder(freq_features)
            else:
                freq_embeds = self._get_frequency_encoding(batch_size, device)
            modality_features.append(freq_embeds)
        
        # 4. 多模态融合
        fused_features = torch.cat(modality_features, dim=-1)
        enhanced_features = self.fusion_network(fused_features)  # [B, embed_dim]
        
        # 5. 学习注意力权重
        attention_weights = self.attention_weights(enhanced_features)  # [B, 3]
        
        # 6. 加权融合
        weighted_user = user_embeds * attention_weights[:, 0:1]
        if self.use_time_encoding:
            weighted_time = modality_features[1] * attention_weights[:, 1:2]
            enhanced_features = enhanced_features + weighted_time
        if self.use_frequency_encoding:
            idx = 2 if self.use_time_encoding else 1
            weighted_freq = modality_features[idx] * attention_weights[:, 2:3]
            enhanced_features = enhanced_features + weighted_freq
        
        enhanced_features = enhanced_features + weighted_user
        
        return enhanced_features.unsqueeze(1)  # [B, 1, embed_dim]

class HierarchicalConditionEncoder(nn.Module):
    """层次化条件编码器 - 从粗粒度到细粒度"""
    
    def __init__(self, num_users: int, embed_dim: int = 512, 
                 num_hierarchy_levels: int = 3, dropout: float = 0.1):
        super().__init__()
        self.num_users = num_users
        self.embed_dim = embed_dim
        self.num_levels = num_hierarchy_levels
        
        # 用户嵌入
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        
        # 层次化编码器
        self.hierarchy_encoders = nn.ModuleList()
        for level in range(num_hierarchy_levels):
            # 每一层的特征维度递减
            level_dim = embed_dim // (2 ** level)
            encoder = nn.Sequential(
                nn.Linear(embed_dim, level_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(level_dim, level_dim),
                nn.GELU(),
                nn.Linear(level_dim, embed_dim),
            )
            self.hierarchy_encoders.append(encoder)
        
        # 层次融合
        self.hierarchy_fusion = nn.Sequential(
            nn.Linear(embed_dim * num_hierarchy_levels, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        
        # 层次注意力
        self.hierarchy_attention = nn.MultiheadAttention(
            embed_dim, num_heads=8, dropout=dropout, batch_first=True
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.02)
    
    def forward(self, user_indices: torch.Tensor) -> torch.Tensor:
        """
        层次化条件编码
        
        Args:
            user_indices: [B] 用户索引
            
        Returns:
            [B, num_levels, embed_dim] 层次化条件表示
        """
        batch_size = user_indices.shape[0]
        
        # 基础用户嵌入
        user_embeds = self.user_embedding(user_indices)  # [B, embed_dim]
        
        # 层次化编码
        hierarchy_features = []
        for level, encoder in enumerate(self.hierarchy_encoders):
            level_features = encoder(user_embeds)  # [B, embed_dim]
            hierarchy_features.append(level_features)
        
        # 堆叠层次特征
        stacked_features = torch.stack(hierarchy_features, dim=1)  # [B, num_levels, embed_dim]
        
        # 层次注意力
        attended_features, _ = self.hierarchy_attention(
            stacked_features, stacked_features, stacked_features
        )  # [B, num_levels, embed_dim]
        
        # 残差连接
        final_features = attended_features + stacked_features
        
        return final_features

# 使用示例
if __name__ == "__main__":
    # 测试多模态编码器
    multimodal_encoder = MultiModalConditionEncoder(
        num_users=31,
        embed_dim=512,
        use_time_encoding=True,
        use_frequency_encoding=True
    )
    
    user_ids = torch.tensor([0, 5, 10])
    output = multimodal_encoder(user_ids)
    print(f"Multimodal output shape: {output.shape}")  # [3, 1, 512]
    
    # 测试层次化编码器
    hierarchical_encoder = HierarchicalConditionEncoder(
        num_users=31,
        embed_dim=512,
        num_hierarchy_levels=3
    )
    
    hierarchical_output = hierarchical_encoder(user_ids)
    print(f"Hierarchical output shape: {hierarchical_output.shape}")  # [3, 3, 512]
