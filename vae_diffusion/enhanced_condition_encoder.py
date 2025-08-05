#!/usr/bin/env python3
"""
增强的用户条件编码器
专门针对微多普勒图像的微妙用户差异设计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EnhancedUserConditionEncoder(nn.Module):
    """增强的用户条件编码器 - 专门处理微妙的用户差异"""
    
    def __init__(self, num_users: int, embed_dim: int = 512, 
                 num_tokens: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_users = num_users
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        
        # 1. 多层用户嵌入
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        
        # 2. 层次化特征提取
        self.feature_extractor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        
        # 3. 用户特征放大器 - 增强微妙差异
        self.amplifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        
        # 4. 多token扩展 - 增加表达能力
        self.token_expander = nn.Linear(embed_dim, embed_dim * num_tokens)
        
        # 5. 位置编码 - 为多token添加位置信息
        self.position_embedding = nn.Parameter(
            torch.randn(num_tokens, embed_dim) * 0.02
        )
        
        # 6. 可学习的用户特异性缩放因子
        self.user_scale_factors = nn.Parameter(
            torch.ones(num_users, 1) * 2.0
        )
        
        # 7. 注意力机制 - 自适应重要性权重
        self.self_attention = nn.MultiheadAttention(
            embed_dim, num_heads=8, dropout=dropout, batch_first=True
        )
        
        # 8. 最终投影层
        self.final_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        nn.init.normal_(self.user_embedding.weight, std=0.02)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, user_indices: torch.Tensor) -> torch.Tensor:
        """
        增强的用户条件编码
        
        Args:
            user_indices: [B] 用户索引
            
        Returns:
            [B, num_tokens, embed_dim] 增强的用户表示
        """
        batch_size = user_indices.shape[0]
        device = user_indices.device
        
        # 1. 基础用户嵌入
        user_embeds = self.user_embedding(user_indices)  # [B, embed_dim]
        
        # 2. 层次化特征提取
        enhanced_features = self.feature_extractor(user_embeds)  # [B, embed_dim]
        
        # 3. 特征放大 - 增强微妙差异
        amplified_features = self.amplifier(enhanced_features)  # [B, embed_dim]
        
        # 4. 应用用户特异性缩放
        user_scales = self.user_scale_factors[user_indices]  # [B, 1]
        scaled_features = amplified_features * user_scales  # [B, embed_dim]
        
        # 5. 扩展到多个token
        expanded_features = self.token_expander(scaled_features)  # [B, embed_dim * num_tokens]
        multi_token_features = expanded_features.view(
            batch_size, self.num_tokens, self.embed_dim
        )  # [B, num_tokens, embed_dim]
        
        # 6. 添加位置编码
        position_embeds = self.position_embedding.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [B, num_tokens, embed_dim]
        positioned_features = multi_token_features + position_embeds
        
        # 7. 自注意力 - 学习token间的关系
        attended_features, _ = self.self_attention(
            positioned_features, positioned_features, positioned_features
        )  # [B, num_tokens, embed_dim]
        
        # 8. 残差连接
        attended_features = attended_features + positioned_features
        
        # 9. 最终投影
        final_features = self.final_proj(attended_features)  # [B, num_tokens, embed_dim]
        
        return final_features

class AdaptiveUserConditionEncoder(nn.Module):
    """自适应用户条件编码器 - 根据用户差异动态调整"""
    
    def __init__(self, num_users: int, embed_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.num_users = num_users
        self.embed_dim = embed_dim
        
        # 用户嵌入
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        
        # 用户差异度学习器
        self.difference_learner = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 条件强度调节器
        self.intensity_modulator = nn.Sequential(
            nn.Linear(embed_dim + 1, embed_dim * 2),  # +1 for difference score
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        
        # 多尺度特征提取
        self.multi_scale_features = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ) for _ in range(3)  # 3个不同尺度
        ])
        
        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.02)
    
    def forward(self, user_indices: torch.Tensor) -> torch.Tensor:
        """
        自适应用户条件编码
        
        Args:
            user_indices: [B] 用户索引
            
        Returns:
            [B, 1, embed_dim] 自适应用户表示
        """
        # 基础嵌入
        user_embeds = self.user_embedding(user_indices)  # [B, embed_dim]
        
        # 学习用户差异度
        difference_score = self.difference_learner(user_embeds)  # [B, 1]
        
        # 多尺度特征提取
        multi_scale_feats = []
        for scale_extractor in self.multi_scale_features:
            scale_feat = scale_extractor(user_embeds)
            multi_scale_feats.append(scale_feat)
        
        # 特征融合
        fused_features = torch.cat(multi_scale_feats, dim=-1)  # [B, embed_dim * 3]
        fused_features = self.feature_fusion(fused_features)  # [B, embed_dim]
        
        # 强度调节
        intensity_input = torch.cat([fused_features, difference_score], dim=-1)
        modulated_features = self.intensity_modulator(intensity_input)  # [B, embed_dim]
        
        return modulated_features.unsqueeze(1)  # [B, 1, embed_dim]

# 使用示例
if __name__ == "__main__":
    # 测试增强条件编码器
    encoder = EnhancedUserConditionEncoder(
        num_users=31,
        embed_dim=512,
        num_tokens=8
    )
    
    user_ids = torch.tensor([0, 5, 10])
    output = encoder(user_ids)
    print(f"Enhanced output shape: {output.shape}")  # [3, 8, 512]
    
    # 测试自适应条件编码器
    adaptive_encoder = AdaptiveUserConditionEncoder(
        num_users=31,
        embed_dim=512
    )
    
    adaptive_output = adaptive_encoder(user_ids)
    print(f"Adaptive output shape: {adaptive_output.shape}")  # [3, 1, 512]
