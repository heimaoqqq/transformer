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
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

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
        
        # 用户ID嵌入
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        
        # 可学习的用户特征增强
        self.user_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )
        
        # 初始化
        nn.init.normal_(self.user_embedding.weight, std=0.02)
    
    def forward(self, user_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            user_ids: [batch_size] 用户ID
        Returns:
            user_embeds: [batch_size, embed_dim] 用户嵌入
        """
        user_embeds = self.user_embedding(user_ids)
        user_embeds = self.user_mlp(user_embeds)
        return user_embeds

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
        
        # 用户条件编码器
        self.user_encoder = UserConditionEncoder(
            num_users=num_users,
            embed_dim=n_embd,
            dropout=dropout,
        )
        
        # 配置GPT模型
        config = GPT2Config(
            vocab_size=vocab_size + 1,  # +1 for special tokens
            n_positions=max_seq_len + 1,  # +1 for user token
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            use_cache=False,
            add_cross_attention=use_cross_attention,
        )
        
        # 创建GPT模型
        self.transformer = GPT2LMHeadModel(config)
        
        # 特殊token
        self.user_token_id = vocab_size  # 用户token ID
        self.pad_token_id = vocab_size   # padding token
        
        # 如果使用交叉注意力，需要投影层
        if use_cross_attention:
            self.user_proj = nn.Linear(n_embd, n_embd)
        
        print(f"🤖 微多普勒Transformer初始化:")
        print(f"   词汇表大小: {vocab_size}")
        print(f"   序列长度: {max_seq_len}")
        print(f"   用户数量: {num_users}")
        print(f"   嵌入维度: {n_embd}")
        print(f"   Transformer层数: {n_layer}")
        print(f"   注意力头数: {n_head}")
        print(f"   交叉注意力: {use_cross_attention}")
        
        # 计算参数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"   总参数量: {total_params/1e6:.1f}M")
    
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
            
            # 输入序列：[user_token] + [token1, token2, ..., token_n-1]
            user_tokens = torch.full((batch_size, 1), self.user_token_id, device=device)
            input_ids = torch.cat([user_tokens, token_sequences[:, :-1]], dim=1)
            
            # 目标序列：[user_token] + [token1, token2, ..., token_n]
            labels = torch.cat([user_tokens, token_sequences], dim=1)
            
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
            # 交叉注意力模式：用户嵌入作为encoder_hidden_states
            encoder_hidden_states = self.user_proj(user_embeds).unsqueeze(1)  # [batch_size, 1, n_embd]
            encoder_attention_mask = torch.ones(batch_size, 1, device=device)
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
