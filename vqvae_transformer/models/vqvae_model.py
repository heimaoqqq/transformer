#!/usr/bin/env python3
"""
防坍缩VQ-VAE模型实现
基于diffusers VQModel，增加EMA更新、码本监控等防坍缩机制
针对微多普勒时频图优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass

# 导入diffusers - 统一环境保证可用
from diffusers.models.autoencoders.vq_model import VQModel
from diffusers.models.autoencoders.vq_model import VectorQuantizer

@dataclass
class VQVAEOutput:
    """VQ-VAE输出类，兼容diffusers格式"""
    sample: torch.Tensor
    vq_loss: torch.Tensor
    encoding_indices: Optional[torch.Tensor] = None
    latents: Optional[torch.Tensor] = None

class EMAVectorQuantizer(nn.Module):
    """
    带EMA更新的向量量化器，防止码本坍缩
    """
    
    def __init__(
        self,
        n_embed: int = 1024,
        embed_dim: int = 256,
        beta: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
        restart_threshold: float = 1.0,
    ):
        super().__init__()
        
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.beta = beta
        self.decay = decay
        self.eps = eps
        self.restart_threshold = restart_threshold
        
        # 码本嵌入
        self.embedding = nn.Embedding(n_embed, embed_dim)
        self.embedding.weight.data.uniform_(-1/n_embed, 1/n_embed)
        
        # EMA参数 (不参与梯度更新)
        self.register_buffer('ema_count', torch.zeros(n_embed))
        self.register_buffer('ema_weight', self.embedding.weight.data.clone())
        
        # 监控参数
        self.register_buffer('usage_count', torch.zeros(n_embed))  # 累积使用次数
        self.register_buffer('total_updates', torch.tensor(0))

        # Epoch级别统计
        self.register_buffer('epoch_usage_mask', torch.zeros(n_embed, dtype=torch.bool))  # 当前epoch是否使用
        self.register_buffer('epoch_usage_count', torch.zeros(n_embed))  # 当前epoch使用次数
    
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        Args:
            inputs: [B, C, H, W] 输入特征
        Returns:
            quantized: 量化后的特征
            loss: VQ损失
            encoding_indices: 编码索引
        """
        # 展平输入 [B, C, H, W] -> [B*H*W, C]
        input_shape = inputs.shape
        flat_input = inputs.contiguous().view(-1, self.embed_dim)
        
        # 计算距离
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        
        # 找到最近的码本向量
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.n_embed, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # 量化
        quantized = torch.matmul(encodings, self.embedding.weight).contiguous().view(input_shape)
        
        # 更新EMA (仅在训练时)
        if self.training:
            self._update_ema(encodings, flat_input)
            self._update_usage_stats(encoding_indices)
        
        # 计算损失
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.beta * e_latent_loss
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        # 保持2D形状 [B, H, W] 用于Transformer训练
        encoding_indices_2d = encoding_indices.contiguous().view(input_shape[0], input_shape[2], input_shape[3])
        return quantized, loss, encoding_indices_2d
    
    def _update_ema(self, encodings: torch.Tensor, flat_input: torch.Tensor):
        """更新EMA参数"""
        with torch.no_grad():
            # 更新计数
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, 0)
            
            # 更新权重
            n = torch.sum(encodings, 0, keepdim=True)
            self.ema_weight = (self.decay * self.ema_weight + 
                              (1 - self.decay) * torch.matmul(encodings.t(), flat_input))
            
            # 归一化并更新嵌入
            normalized_weight = self.ema_weight / (self.ema_count.unsqueeze(1) + self.eps)
            self.embedding.weight.data.copy_(normalized_weight)
            
            # 重置未使用的码本向量
            self._restart_unused_codes()
    
    def _update_usage_stats(self, encoding_indices: torch.Tensor):
        """更新使用统计"""
        with torch.no_grad():
            unique_indices = torch.unique(encoding_indices)

            # 累积统计 (保留原有逻辑)
            self.usage_count[unique_indices] += 1
            self.total_updates += 1

            # Epoch统计 (新增)
            self.epoch_usage_mask[unique_indices] = True
            self.epoch_usage_count[unique_indices] += 1
    
    def _restart_unused_codes(self):
        """重置未使用的码本向量"""
        if self.total_updates > 0 and self.total_updates % 1000 == 0:  # 每1000次更新检查一次
            with torch.no_grad():
                # 找到使用频率低的码本向量
                usage_freq = self.ema_count / (self.total_updates + self.eps)
                unused_mask = usage_freq < self.restart_threshold / self.n_embed
                
                if unused_mask.sum() > 0:
                    print(f"重置 {unused_mask.sum()} 个未使用的码本向量")
                    
                    # 用随机向量重置
                    n_restart = unused_mask.sum()
                    self.embedding.weight.data[unused_mask] = torch.randn(
                        n_restart, self.embed_dim, device=self.embedding.weight.device
                    ) * 0.1
                    
                    # 重置EMA参数
                    self.ema_count[unused_mask] = 0
                    self.ema_weight[unused_mask] = self.embedding.weight.data[unused_mask].clone()
    
    def get_codebook_usage(self) -> Dict[str, float]:
        """获取码本使用统计"""
        with torch.no_grad():
            # 累积统计
            cumulative_active = (self.usage_count > 0).sum().item()
            cumulative_rate = cumulative_active / self.n_embed
            cumulative_entropy = self._compute_entropy(self.usage_count)

            # 当前epoch统计
            epoch_active = self.epoch_usage_mask.sum().item()
            epoch_rate = epoch_active / self.n_embed
            epoch_entropy = self._compute_entropy(self.epoch_usage_count)

            return {
                # Epoch级别统计 (主要关注)
                'epoch_active_codes': epoch_active,
                'epoch_usage_rate': epoch_rate,
                'epoch_entropy': epoch_entropy,

                # 累积统计 (参考)
                'cumulative_active_codes': cumulative_active,
                'cumulative_usage_rate': cumulative_rate,
                'cumulative_entropy': cumulative_entropy,

                # 通用信息
                'total_codes': self.n_embed,
                'total_updates': self.total_updates.item(),

                # 兼容性 (保持原有接口)
                'active_codes': epoch_active,  # 默认返回epoch统计
                'usage_rate': epoch_rate,      # 默认返回epoch统计
                'usage_entropy': epoch_entropy, # 默认返回epoch统计
            }
    
    def _compute_entropy(self, counts: torch.Tensor) -> float:
        """
        计算使用分布的熵
        修复: 考虑完整的概率分布，包括未使用的码本
        """
        counts = counts.float()
        total = counts.sum()
        if total == 0:
            return 0.0

        # 计算所有码本的概率，包括未使用的(概率为0)
        probs = counts / total

        # 只对非零概率计算log，避免log(0)
        log_probs = torch.zeros_like(probs)
        mask = probs > 0
        log_probs[mask] = torch.log(probs[mask])

        # 计算熵: H = -Σ p(x) * log(p(x))
        # 对于p(x)=0的项，p(x)*log(p(x))=0 (约定0*log(0)=0)
        entropy = -(probs * log_probs).sum().item()
        return entropy

    def reset_epoch_stats(self):
        """重置epoch级别的统计信息"""
        with torch.no_grad():
            self.epoch_usage_mask.zero_()
            self.epoch_usage_count.zero_()

    def plot_usage_distribution(self, save_path: Optional[str] = None):
        """可视化码本使用分布"""
        usage = self.usage_count.cpu().numpy()
        
        plt.figure(figsize=(12, 4))
        
        # 使用分布直方图
        plt.subplot(1, 2, 1)
        plt.hist(usage[usage > 0], bins=50, alpha=0.7)
        plt.xlabel('Usage Count')
        plt.ylabel('Number of Codes')
        plt.title('Codebook Usage Distribution')
        
        # 使用频率排序
        plt.subplot(1, 2, 2)
        sorted_usage = np.sort(usage)[::-1]
        plt.plot(sorted_usage)
        plt.xlabel('Code Index (sorted)')
        plt.ylabel('Usage Count')
        plt.title('Codebook Usage (Sorted)')
        plt.yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


class MicroDopplerVQVAE(VQModel):
    """
    针对微多普勒时频图优化的VQ-VAE模型
    基于diffusers VQModel，替换量化器为EMA版本
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types: Tuple[str] = ("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels: Tuple[int] = (128, 256, 512),
        layers_per_block: int = 2,
        act_fn: str = "silu",
        latent_channels: int = 256,
        sample_size: int = 128,
        num_vq_embeddings: int = 1024,
        norm_num_groups: int = 32,
        vq_embed_dim: Optional[int] = None,
        scaling_factor: float = 0.18215,
        # VQ-VAE特定参数
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        restart_threshold: float = 1.0,
    ):
        # 使用父类初始化基础结构
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            latent_channels=latent_channels,
            sample_size=sample_size,
            num_vq_embeddings=num_vq_embeddings,
            norm_num_groups=norm_num_groups,
            vq_embed_dim=vq_embed_dim,
            scaling_factor=scaling_factor,
        )
        
        # 替换量化器为EMA版本
        embed_dim = vq_embed_dim if vq_embed_dim is not None else latent_channels
        self.quantize = EMAVectorQuantizer(
            n_embed=num_vq_embeddings,
            embed_dim=embed_dim,
            beta=commitment_cost,
            decay=ema_decay,
            restart_threshold=restart_threshold,
        )
        
        print(f"🎯 微多普勒VQ-VAE初始化:")
        print(f"   码本大小: {num_vq_embeddings}")
        print(f"   嵌入维度: {embed_dim}")
        print(f"   EMA衰减: {ema_decay}")
        print(f"   Commitment权重: {commitment_cost}")
    
    def encode(self, x: torch.Tensor, return_dict: bool = True):
        """编码为离散tokens"""
        h = self.encoder(x)
        h = self.quant_conv(h)
        quantized, vq_loss, encoding_indices = self.quantize(h)
        
        if not return_dict:
            return (quantized, vq_loss, encoding_indices)
        
        return {
            'latents': quantized,
            'vq_loss': vq_loss,
            'encoding_indices': encoding_indices,
        }
    
    def decode(self, h: torch.Tensor, force_not_quantize: bool = False):
        """从潜在表示解码"""
        if not force_not_quantize:
            quantized, _, _ = self.quantize(h)
            h = quantized
        
        h = self.post_quant_conv(h)
        dec = self.decoder(h)
        return dec
    
    def forward(self, sample: torch.Tensor, return_dict: bool = True):
        """完整的前向传播"""
        encode_result = self.encode(sample, return_dict=True)
        dec = self.decode(encode_result['latents'])

        if not return_dict:
            return (dec, encode_result['vq_loss'])

        return VQVAEOutput(
            sample=dec,
            vq_loss=encode_result['vq_loss'],
            encoding_indices=encode_result['encoding_indices'],
            latents=encode_result['latents'],
        )
    
    def get_codebook_stats(self) -> Dict[str, float]:
        """获取码本统计信息"""
        return self.quantize.get_codebook_usage()
    
    def plot_codebook_usage(self, save_path: Optional[str] = None):
        """可视化码本使用情况"""
        self.quantize.plot_usage_distribution(save_path)
    
    def reset_codebook_stats(self):
        """重置码本统计"""
        self.quantize.usage_count.zero_()
        self.quantize.total_updates.zero_()
        self.quantize.reset_epoch_stats()

    def reset_epoch_stats(self):
        """重置epoch级别统计"""
        self.quantize.reset_epoch_stats()
