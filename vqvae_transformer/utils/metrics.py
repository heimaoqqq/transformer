#!/usr/bin/env python3
"""
评估指标
复用主项目的评估指标，适配VQ-VAE + Transformer需求
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, Tuple
import sys
from pathlib import Path

# 添加主项目路径以复用评估指标
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from utils.metrics import calculate_psnr as base_calculate_psnr
    from utils.metrics import calculate_ssim as base_calculate_ssim
except ImportError:
    # 如果无法导入，提供基础实现
    def base_calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
        """计算PSNR"""
        mse = F.mse_loss(img1, img2)
        if mse == 0:
            return float('inf')
        return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()
    
    def base_calculate_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
        """简化的SSIM计算"""
        # 这是一个简化版本，实际应该使用更完整的SSIM实现
        mu1 = img1.mean()
        mu2 = img2.mean()
        sigma1 = img1.var()
        sigma2 = img2.var()
        sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))
        
        return ssim.item()

def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    计算PSNR
    Args:
        img1: 第一张图像 [C, H, W] 或 [H, W]
        img2: 第二张图像 [C, H, W] 或 [H, W]
    Returns:
        psnr: PSNR值
    """
    return base_calculate_psnr(img1, img2)

def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    计算SSIM
    Args:
        img1: 第一张图像 [C, H, W] 或 [H, W]
        img2: 第二张图像 [C, H, W] 或 [H, W]
    Returns:
        ssim: SSIM值
    """
    return base_calculate_ssim(img1, img2)

def calculate_lpips(img1: torch.Tensor, img2: torch.Tensor, lpips_model=None) -> float:
    """
    计算LPIPS感知距离
    Args:
        img1: 第一张图像 [C, H, W]
        img2: 第二张图像 [C, H, W]
        lpips_model: LPIPS模型
    Returns:
        lpips: LPIPS距离
    """
    if lpips_model is None:
        try:
            import lpips
            lpips_model = lpips.LPIPS(net='alex')
        except ImportError:
            print("⚠️ LPIPS未安装，返回0")
            return 0.0
    
    # 确保输入在正确范围内
    img1 = img1.unsqueeze(0) if img1.dim() == 3 else img1
    img2 = img2.unsqueeze(0) if img2.dim() == 3 else img2
    
    # LPIPS期望输入在[-1, 1]范围内
    if img1.max() <= 1.0:
        img1 = img1 * 2.0 - 1.0
        img2 = img2 * 2.0 - 1.0
    
    with torch.no_grad():
        distance = lpips_model(img1, img2)
    
    return distance.item()

def calculate_fid(real_features: torch.Tensor, fake_features: torch.Tensor) -> float:
    """
    计算FID (Fréchet Inception Distance)
    Args:
        real_features: 真实图像特征 [N, D]
        fake_features: 生成图像特征 [N, D]
    Returns:
        fid: FID分数
    """
    # 计算均值和协方差
    mu1 = real_features.mean(dim=0)
    mu2 = fake_features.mean(dim=0)
    
    sigma1 = torch.cov(real_features.T)
    sigma2 = torch.cov(fake_features.T)
    
    # 计算FID
    diff = mu1 - mu2
    covmean = torch.sqrt(sigma1 @ sigma2)
    
    if torch.isnan(covmean).any():
        covmean = torch.sqrt((sigma1 + sigma2) / 2)
    
    fid = diff.dot(diff) + torch.trace(sigma1 + sigma2 - 2 * covmean)
    
    return fid.item()

def calculate_is(features: torch.Tensor, splits: int = 10) -> Tuple[float, float]:
    """
    计算Inception Score
    Args:
        features: 图像特征 [N, num_classes]
        splits: 分割数量
    Returns:
        is_mean: IS均值
        is_std: IS标准差
    """
    # 计算条件概率和边际概率
    p_yx = F.softmax(features, dim=1)
    p_y = p_yx.mean(dim=0, keepdim=True)
    
    # 计算KL散度
    kl_div = p_yx * (torch.log(p_yx) - torch.log(p_y))
    kl_div = kl_div.sum(dim=1)
    
    # 分割计算
    n_samples = features.size(0)
    split_size = n_samples // splits
    
    scores = []
    for i in range(splits):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i < splits - 1 else n_samples
        
        split_kl = kl_div[start_idx:end_idx].mean()
        scores.append(torch.exp(split_kl).item())
    
    return np.mean(scores), np.std(scores)

class VQVAEMetrics:
    """VQ-VAE专用评估指标"""
    
    def __init__(self, device="cuda"):
        self.device = device
        self.lpips_model = None
        
        # 尝试加载LPIPS模型
        try:
            import lpips
            self.lpips_model = lpips.LPIPS(net='alex').to(device)
            print("✅ LPIPS模型加载成功")
        except ImportError:
            print("⚠️ LPIPS未安装，将跳过感知损失计算")
    
    def evaluate_reconstruction(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
    ) -> dict:
        """
        评估重建质量
        Args:
            original: 原始图像 [B, C, H, W]
            reconstructed: 重建图像 [B, C, H, W]
        Returns:
            metrics: 评估指标字典
        """
        metrics = {
            'psnr': [],
            'ssim': [],
            'lpips': [],
        }
        
        batch_size = original.size(0)
        
        for i in range(batch_size):
            # PSNR和SSIM
            psnr = calculate_psnr(original[i], reconstructed[i])
            ssim = calculate_ssim(original[i], reconstructed[i])
            
            metrics['psnr'].append(psnr)
            metrics['ssim'].append(ssim)
            
            # LPIPS
            if self.lpips_model is not None:
                lpips_score = calculate_lpips(
                    original[i], reconstructed[i], self.lpips_model
                )
                metrics['lpips'].append(lpips_score)
        
        # 计算平均值
        result = {
            'psnr_mean': np.mean(metrics['psnr']),
            'psnr_std': np.std(metrics['psnr']),
            'ssim_mean': np.mean(metrics['ssim']),
            'ssim_std': np.std(metrics['ssim']),
        }
        
        if metrics['lpips']:
            result.update({
                'lpips_mean': np.mean(metrics['lpips']),
                'lpips_std': np.std(metrics['lpips']),
            })
        
        return result
    
    def evaluate_codebook_usage(self, vqvae_model) -> dict:
        """
        评估码本使用情况
        Args:
            vqvae_model: VQ-VAE模型
        Returns:
            usage_stats: 使用统计
        """
        return vqvae_model.get_codebook_stats()

class TransformerMetrics:
    """Transformer专用评估指标"""
    
    def __init__(self):
        pass
    
    def calculate_perplexity(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        """
        计算困惑度
        Args:
            logits: 模型输出 [B, L, V]
            targets: 目标序列 [B, L]
        Returns:
            perplexity: 困惑度
        """
        # 计算交叉熵损失
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction='mean'
        )
        
        # 困惑度 = exp(loss)
        perplexity = torch.exp(loss)
        
        return perplexity.item()
    
    def calculate_token_accuracy(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        """
        计算token级准确率
        Args:
            logits: 模型输出 [B, L, V]
            targets: 目标序列 [B, L]
        Returns:
            accuracy: 准确率
        """
        predictions = torch.argmax(logits, dim=-1)
        correct = (predictions == targets).float()
        accuracy = correct.mean()
        
        return accuracy.item()
    
    def evaluate_generation_diversity(self, generated_sequences: list) -> dict:
        """
        评估生成多样性
        Args:
            generated_sequences: 生成的序列列表
        Returns:
            diversity_metrics: 多样性指标
        """
        if len(generated_sequences) < 2:
            return {'diversity': 0.0, 'uniqueness': 0.0}
        
        # 计算序列间的平均距离
        total_distance = 0
        count = 0
        
        for i in range(len(generated_sequences)):
            for j in range(i + 1, len(generated_sequences)):
                seq1 = generated_sequences[i]
                seq2 = generated_sequences[j]
                
                # 计算编辑距离（简化版）
                distance = (seq1 != seq2).float().mean().item()
                total_distance += distance
                count += 1
        
        diversity = total_distance / count if count > 0 else 0.0
        
        # 计算唯一序列比例
        unique_sequences = len(set(tuple(seq.tolist()) for seq in generated_sequences))
        uniqueness = unique_sequences / len(generated_sequences)
        
        return {
            'diversity': diversity,
            'uniqueness': uniqueness,
            'unique_count': unique_sequences,
            'total_count': len(generated_sequences),
        }
