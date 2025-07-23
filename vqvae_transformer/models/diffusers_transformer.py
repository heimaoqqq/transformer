#!/usr/bin/env python3
"""
使用diffusers库的标准Transformer实现
基于成熟的、经过验证的diffusers组件
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

try:
    from diffusers import Transformer2DModel
    from diffusers.configuration_utils import ConfigMixin, register_to_config
    from diffusers.models.modeling_utils import ModelMixin
    from diffusers.utils import BaseOutput
    DIFFUSERS_AVAILABLE = True
except ImportError:
    print("❌ diffusers库不可用，请安装: pip install diffusers")
    DIFFUSERS_AVAILABLE = False
    # 创建兼容的基类
    class ModelMixin(nn.Module):
        pass
    class ConfigMixin:
        pass
    def register_to_config(func):
        return func
    class BaseOutput:
        pass

@dataclass
class TransformerOutput(BaseOutput):
    """Transformer输出"""
    logits: torch.FloatTensor
    loss: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class DiffusersTransformerModel(ModelMixin, ConfigMixin):
    """
    基于diffusers的标准Transformer模型
    用于VQ-VAE token序列生成
    """
    
    @register_to_config
    def __init__(
        self,
        vocab_size: int = 1024,
        max_seq_len: int = 1024,
        num_users: int = 31,
        num_layers: int = 8,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        in_channels: int = None,  # 将从vocab_size推导
        cross_attention_dim: int = 512,
        dropout: float = 0.1,
        activation_fn: str = "gelu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = True,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        attention_type: str = "default",
    ):
        super().__init__()
        
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("diffusers库不可用，请安装: pip install diffusers")
        
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.num_users = num_users
        
        # 计算内部维度
        inner_dim = num_attention_heads * attention_head_dim
        if in_channels is None:
            in_channels = inner_dim
        
        # Token嵌入层
        self.token_embedding = nn.Embedding(vocab_size, inner_dim)
        
        # 用户嵌入层
        self.user_embedding = nn.Embedding(num_users + 1, inner_dim)  # +1 for padding
        
        # 位置嵌入
        self.position_embedding = nn.Embedding(max_seq_len, inner_dim)
        
        # 使用diffusers的标准Transformer2D模型
        self.transformer = Transformer2DModel(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            num_layers=num_layers,
            dropout=dropout,
            cross_attention_dim=cross_attention_dim,
            activation_fn=activation_fn,
            num_embeds_ada_norm=num_embeds_ada_norm,
            attention_bias=attention_bias,
            only_cross_attention=only_cross_attention,
            double_self_attention=double_self_attention,
            upcast_attention=upcast_attention,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            attention_type=attention_type,
        )
        
        # 输出投影层
        self.output_projection = nn.Linear(inner_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        user_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: bool = True,
    ) -> TransformerOutput:
        """
        前向传播
        
        Args:
            input_ids: Token序列 [batch_size, seq_len]
            user_ids: 用户ID [batch_size]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            labels: 标签（用于计算损失） [batch_size, seq_len]
            return_dict: 是否返回字典格式
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token嵌入
        token_embeds = self.token_embedding(input_ids)  # [B, L, D]
        
        # 用户嵌入
        user_embeds = self.user_embedding(user_ids).unsqueeze(1)  # [B, 1, D]
        user_embeds = user_embeds.expand(-1, seq_len, -1)  # [B, L, D]
        
        # 位置嵌入
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embedding(position_ids)  # [B, L, D]
        
        # 组合嵌入
        hidden_states = token_embeds + user_embeds + position_embeds
        hidden_states = self.dropout(hidden_states)
        
        # 重塑为2D格式 (diffusers Transformer2D期望的格式)
        # 假设序列长度是32x32=1024，重塑为32x32
        height = width = int(seq_len ** 0.5)
        if height * width != seq_len:
            # 如果不是完全平方数，使用最接近的矩形
            height = 32
            width = seq_len // height
        
        hidden_states = hidden_states.permute(0, 2, 1)  # [B, D, L]
        hidden_states = hidden_states.view(batch_size, -1, height, width)  # [B, D, H, W]
        
        # 通过Transformer
        transformer_output = self.transformer(
            hidden_states,
            encoder_hidden_states=None,  # 不使用交叉注意力
            return_dict=True,
        )
        
        # 获取输出并重塑回序列格式
        hidden_states = transformer_output.sample  # [B, D, H, W]
        hidden_states = hidden_states.view(batch_size, -1, seq_len)  # [B, D, L]
        hidden_states = hidden_states.permute(0, 2, 1)  # [B, L, D]
        
        # 输出投影
        logits = self.output_projection(hidden_states)  # [B, L, V]
        
        # 计算损失
        loss = None
        if labels is not None:
            # 移位标签用于语言建模
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # 计算交叉熵损失
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1)
            )
        
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output
        
        return TransformerOutput(
            logits=logits,
            loss=loss,
            hidden_states=None,
            attentions=None,
        )
    
    @torch.no_grad()
    def generate(
        self,
        user_ids: torch.LongTensor,
        max_length: int = 1024,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> torch.LongTensor:
        """
        生成token序列
        
        Args:
            user_ids: 用户ID [batch_size]
            max_length: 最大生成长度
            temperature: 采样温度
            top_k: top-k采样
            top_p: top-p采样
            do_sample: 是否采样
        
        Returns:
            生成的token序列 [batch_size, max_length]
        """
        batch_size = user_ids.shape[0]
        device = user_ids.device
        
        # 初始化序列（从特殊的开始token开始）
        generated = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        
        for _ in range(max_length - 1):
            # 前向传播
            outputs = self.forward(
                input_ids=generated,
                user_ids=user_ids,
                return_dict=True,
            )
            
            # 获取下一个token的logits
            next_token_logits = outputs.logits[:, -1, :] / temperature
            
            # 过滤词汇表外的token
            next_token_logits[:, self.vocab_size:] = -float('inf')
            
            if do_sample:
                # Top-k采样
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, -float('inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Top-p采样
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # 移除累积概率超过top_p的token
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # 采样
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # 贪婪解码
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # 添加到序列
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated
