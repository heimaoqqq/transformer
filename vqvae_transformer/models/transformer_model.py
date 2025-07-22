#!/usr/bin/env python3
"""
条件Transformer模型实现
用于从用户ID生成VQ-VAE token序列
针对微多普勒时频图的用户特征生成优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict

# 条件导入transformers - 在VQ-VAE环境中可能不可用
try:
    from transformers import GPT2Config, GPT2LMHeadModel
    from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️ transformers不可用，Transformer模型将不可用")
    TRANSFORMERS_AVAILABLE = False
    # 创建兼容的基类
    class GPT2Config:
        def __init__(self, *args, **kwargs):
            pass

    class GPT2LMHeadModel(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    class CausalLMOutputWithCrossAttentions:
        def __init__(self, *args, **kwargs):
            pass

class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]

class UserConditionEncoder(nn.Module):
    """用户条件编码器"""
    
    def __init__(
        self,
        num_users: int = 31,
        embed_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_users = num_users
        self.embed_dim = embed_dim
        
        # 用户ID嵌入 - 支持用户ID从1开始
        self.user_embedding = nn.Embedding(num_users + 1, embed_dim)
        
        # 增强的用户特征学习网络 - 专为微小差异设计
        self.user_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),  # 更大的隐藏层
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),  # 添加LayerNorm稳定训练
        )

        # 用户特征多头注意力 - 增强特征表达能力
        self.user_self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 初始化
        nn.init.normal_(self.user_embedding.weight, std=0.02)
    
    def forward(self, user_ids: torch.Tensor) -> torch.Tensor:
        """
        增强的用户特征编码
        Args:
            user_ids: [batch_size] 用户ID
        Returns:
            user_embeds: [batch_size, embed_dim] 增强的用户嵌入
        """
        # 基础用户嵌入
        user_embeds = self.user_embedding(user_ids)  # [B, embed_dim]

        # 通过MLP增强特征
        enhanced_embeds = self.user_mlp(user_embeds)  # [B, embed_dim]

        # 自注意力进一步增强用户特征表达
        # 为了使用多头注意力，我们需要序列维度
        user_seq = enhanced_embeds.unsqueeze(1)  # [B, 1, embed_dim]
        attended_embeds, _ = self.user_self_attention(
            user_seq, user_seq, user_seq
        )  # [B, 1, embed_dim]

        # 残差连接
        final_embeds = enhanced_embeds + attended_embeds.squeeze(1)

        return final_embeds

class MicroDopplerTransformer(nn.Module):
    """
    微多普勒条件Transformer
    基于GPT架构，支持用户ID条件生成
    """
    
    def __init__(
        self,
        vocab_size: int = 1024,          # VQ码本大小
        max_seq_len: int = 256,          # 最大序列长度 (16x16 = 256)
        num_users: int = 31,             # 用户数量
        n_embd: int = 512,               # 嵌入维度
        n_layer: int = 8,                # Transformer层数
        n_head: int = 8,                 # 注意力头数
        dropout: float = 0.1,            # Dropout率
        use_cross_attention: bool = True, # 是否使用交叉注意力
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.num_users = num_users
        self.n_embd = n_embd
        self.use_cross_attention = use_cross_attention

        # 确保在使用前设置扩展因子
        self.user_expansion_factor = 4 if use_cross_attention else 1
        
        # 用户条件编码器
        self.user_encoder = UserConditionEncoder(
            num_users=num_users,
            embed_dim=n_embd,
            dropout=dropout,
        )
        
        # 配置自定义GPT模型 - 专为VQ-VAE视觉token优化
        config = GPT2Config(
            vocab_size=vocab_size + 1,  # VQ-VAE码本大小(1024) + 1个特殊token
            n_positions=max_seq_len + 1,  # 序列长度(1024) + 1个用户token
            n_embd=n_embd,  # 嵌入维度(512)
            n_layer=n_layer,  # Transformer层数(8)
            n_head=n_head,  # 注意力头数(8)
            n_inner=n_embd * 4,  # FFN内部维度(2048)
            activation_function="gelu_new",  # 使用新版GELU
            resid_pdrop=dropout,  # 残差连接dropout
            embd_pdrop=dropout,   # 嵌入层dropout
            attn_pdrop=dropout,   # 注意力dropout
            layer_norm_epsilon=1e-5,  # LayerNorm epsilon
            initializer_range=0.02,   # 权重初始化范围
            use_cache=False,  # 训练时不使用缓存
            add_cross_attention=use_cross_attention,  # 是否添加交叉注意力
            # 确保不加载预训练权重
            _name_or_path="",
        )
        
        # 创建自定义GPT模型（不加载预训练权重）
        self.transformer = GPT2LMHeadModel(config)

        # 重新初始化权重以确保适合视觉token
        self._init_weights()
        
        # 特殊token
        self.user_token_id = vocab_size  # 用户token ID
        self.pad_token_id = vocab_size   # padding token
        
        # 增强的交叉注意力机制
        if use_cross_attention:
            # 多层用户特征投影，增强用户信息表达
            self.user_proj = nn.Sequential(
                nn.Linear(n_embd, n_embd * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(n_embd * 2, n_embd),
                nn.LayerNorm(n_embd),
            )

            # 用户特征扩展 - 从1个token扩展到多个token增强表达能力
            self.user_expansion_factor = 4  # 扩展为4个token
            self.user_expand = nn.Linear(n_embd, n_embd * self.user_expansion_factor)
        
        print(f"🤖 微多普勒Transformer初始化:")
        print(f"   模型类型: 自定义GPT2 (专为视觉token优化)")
        print(f"   词汇表大小: {vocab_size} + 1个特殊token")
        print(f"   序列长度: {max_seq_len} (32×32 token map) + 1个用户token")
        print(f"   用户数量: {num_users}")
        print(f"   嵌入维度: {n_embd}")
        print(f"   Transformer层数: {n_layer}")
        print(f"   注意力头数: {n_head}")
        print(f"   交叉注意力: {use_cross_attention}")
        print(f"   预训练权重: 不使用 (从头训练)")
        
        # 详细的参数量统计
        user_encoder_params = sum(p.numel() for p in self.user_encoder.parameters())
        transformer_params = sum(p.numel() for p in self.transformer.parameters())
        if use_cross_attention:
            user_proj_params = sum(p.numel() for p in self.user_proj.parameters())
            user_expand_params = sum(p.numel() for p in self.user_expand.parameters())
        else:
            user_proj_params = 0
            user_expand_params = 0

        total_params = sum(p.numel() for p in self.parameters())

        print(f"   📊 参数量详细统计:")
        print(f"      用户编码器: {user_encoder_params/1e6:.2f}M")
        print(f"      Transformer主体: {transformer_params/1e6:.2f}M")
        print(f"      用户投影层: {user_proj_params/1e6:.2f}M")
        print(f"      用户扩展层: {user_expand_params/1e6:.2f}M")
        print(f"      总参数量: {total_params/1e6:.1f}M")

    def _init_weights(self):
        """初始化模型权重 - 专为视觉token优化"""
        def _init_module(module):
            if isinstance(module, nn.Linear):
                # 线性层使用正态分布初始化
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                # 嵌入层使用正态分布初始化
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                # LayerNorm使用标准初始化
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.ones_(module.weight)

        # 应用初始化到所有模块
        self.apply(_init_module)
        print("✅ 模型权重已重新初始化（专为视觉token优化）")

        # 验证增强功能是否正确启用
        self._verify_enhancements()

    def _verify_enhancements(self):
        """验证增强功能是否正确启用"""
        print(f"🔍 验证模型增强功能:")

        # 检查用户编码器
        user_mlp_layers = len(self.user_encoder.user_mlp)
        print(f"   用户MLP层数: {user_mlp_layers} (应该>6)")

        # 检查自注意力
        has_self_attention = hasattr(self.user_encoder, 'user_self_attention')
        print(f"   用户自注意力: {'✅启用' if has_self_attention else '❌未启用'}")

        # 检查交叉注意力增强
        if self.use_cross_attention:
            has_user_proj = hasattr(self, 'user_proj')
            has_user_expand = hasattr(self, 'user_expand')
            print(f"   增强用户投影: {'✅启用' if has_user_proj else '❌未启用'}")
            print(f"   用户特征扩展: {'✅启用' if has_user_expand else '❌未启用'}")
            print(f"   扩展因子: {self.user_expansion_factor}")

        # 检查GPT2交叉注意力
        gpt2_has_cross_attn = self.transformer.config.add_cross_attention
        print(f"   GPT2交叉注意力: {'✅启用' if gpt2_has_cross_attn else '❌未启用'}")
    
    def prepare_inputs(
        self, 
        user_ids: torch.Tensor, 
        token_sequences: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        准备模型输入
        Args:
            user_ids: [batch_size] 用户ID
            token_sequences: [batch_size, seq_len] token序列 (训练时提供)
            max_length: 生成时的最大长度
        """
        batch_size = user_ids.size(0)
        device = user_ids.device
        
        if token_sequences is not None:
            # 训练模式：构造输入序列
            seq_len = token_sequences.size(1)
            
            # 自回归训练：用户token预测第一个图像token，每个图像token预测下一个
            user_tokens = torch.full((batch_size, 1), self.user_token_id, device=device)

            # 输入序列：[user_token] + [token1, token2, ..., token_n-1]
            input_ids = torch.cat([user_tokens, token_sequences[:, :-1]], dim=1)  # [B, 1024]

            # 目标序列：[token1] + [token2, token3, ..., token_n] (每个位置预测下一个token)
            # 用户token位置预测token1，token1位置预测token2，...，token_n-1位置预测token_n
            labels = token_sequences  # [B, 1024]
            
            # 注意力掩码
            attention_mask = torch.ones_like(input_ids)
            
        else:
            # 生成模式：只有用户token
            max_length = max_length or self.max_seq_len
            input_ids = torch.full((batch_size, 1), self.user_token_id, device=device)
            labels = None
            attention_mask = torch.ones_like(input_ids)
        
        # 用户条件编码
        user_embeds = self.user_encoder(user_ids)  # [batch_size, n_embd]
        
        if self.use_cross_attention:
            # 增强的交叉注意力模式：扩展用户特征表达
            projected_user_embeds = self.user_proj(user_embeds)  # [B, n_embd]

            # 扩展用户特征为多个token以增强表达能力
            expanded_user_features = self.user_expand(projected_user_embeds)  # [B, n_embd * 4]
            expanded_user_features = expanded_user_features.view(
                batch_size, self.user_expansion_factor, self.n_embd
            )  # [B, 4, n_embd]

            encoder_hidden_states = expanded_user_features
            encoder_attention_mask = torch.ones(batch_size, self.user_expansion_factor, device=device)
        else:
            # 自注意力模式：用户嵌入替换用户token的嵌入
            encoder_hidden_states = None
            encoder_attention_mask = None
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'encoder_hidden_states': encoder_hidden_states,
            'encoder_attention_mask': encoder_attention_mask,
            'user_embeds': user_embeds,
        }
    
    def forward(
        self,
        user_ids: torch.Tensor,
        token_sequences: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        """
        前向传播
        Args:
            user_ids: [batch_size] 用户ID
            token_sequences: [batch_size, seq_len] token序列
        """
        inputs = self.prepare_inputs(user_ids, token_sequences)
        
        # 如果不使用交叉注意力，需要手动替换用户token的嵌入
        if not self.use_cross_attention:
            # 获取token嵌入
            inputs_embeds = self.transformer.transformer.wte(inputs['input_ids'])
            
            # 替换用户token位置的嵌入
            user_positions = (inputs['input_ids'] == self.user_token_id)
            inputs_embeds[user_positions] = inputs['user_embeds'].unsqueeze(1).expand(-1, user_positions.sum(1).max(), -1)[user_positions]
            
            # 使用嵌入而不是token ID
            outputs = self.transformer(
                inputs_embeds=inputs_embeds,
                attention_mask=inputs['attention_mask'],
                labels=inputs['labels'],
                return_dict=return_dict,
            )
        else:
            # 使用交叉注意力
            outputs = self.transformer(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                encoder_hidden_states=inputs['encoder_hidden_states'],
                encoder_attention_mask=inputs['encoder_attention_mask'],
                labels=inputs['labels'],
                return_dict=return_dict,
            )
        
        return outputs

    @torch.no_grad()
    def generate(
        self,
        user_ids: torch.Tensor,
        max_length: int = 256,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        num_return_sequences: int = 1,
    ) -> torch.Tensor:
        """
        生成token序列
        Args:
            user_ids: [batch_size] 用户ID
            max_length: 最大生成长度
            temperature: 采样温度
            top_k: top-k采样
            top_p: nucleus采样
            do_sample: 是否采样
            num_return_sequences: 每个用户生成的序列数
        Returns:
            generated_sequences: [batch_size * num_return_sequences, seq_len]
        """
        batch_size = user_ids.size(0)
        device = user_ids.device

        # 扩展用户ID以支持多序列生成
        if num_return_sequences > 1:
            user_ids = user_ids.unsqueeze(1).expand(-1, num_return_sequences).flatten()
            batch_size = user_ids.size(0)

        # 准备初始输入
        inputs = self.prepare_inputs(user_ids, max_length=max_length)
        input_ids = inputs['input_ids']  # [batch_size, 1] 只有用户token

        # 生成循环
        for _ in range(max_length):
            # 前向传播
            if not self.use_cross_attention:
                # 手动处理用户嵌入
                inputs_embeds = self.transformer.transformer.wte(input_ids)
                user_positions = (input_ids == self.user_token_id)
                if user_positions.any():
                    user_embeds = self.user_encoder(user_ids % self.num_users)  # 处理扩展的user_ids
                    inputs_embeds[user_positions] = user_embeds.unsqueeze(1).expand(-1, user_positions.sum(1).max(), -1)[user_positions]

                outputs = self.transformer(
                    inputs_embeds=inputs_embeds,
                    attention_mask=inputs['attention_mask'],
                    return_dict=True,
                )
            else:
                outputs = self.transformer(
                    input_ids=input_ids,
                    attention_mask=inputs['attention_mask'],
                    encoder_hidden_states=inputs['encoder_hidden_states'],
                    encoder_attention_mask=inputs['encoder_attention_mask'],
                    return_dict=True,
                )

            # 获取下一个token的logits
            next_token_logits = outputs.logits[:, -1, :] / temperature

            # 过滤词汇表，只保留有效的VQ token
            next_token_logits[:, self.vocab_size:] = -float('inf')  # 屏蔽特殊token

            # 采样下一个token
            if do_sample:
                # Top-k采样
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('inf')

                # Top-p (nucleus)采样
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('inf')

                # 采样
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                # 贪心解码
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # 添加到序列
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)

            # 更新注意力掩码
            inputs['attention_mask'] = torch.cat([
                inputs['attention_mask'],
                torch.ones(batch_size, 1, device=device)
            ], dim=-1)

        # 移除用户token，只返回生成的VQ token序列
        generated_sequences = input_ids[:, 1:]  # 去掉第一个用户token

        return generated_sequences
