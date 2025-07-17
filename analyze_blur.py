#!/usr/bin/env python3
"""
分析VAE重建模糊问题
提供具体的改进建议
"""

import torch
import numpy as np
from pathlib import Path
from diffusers import AutoencoderKL
from utils.data_loader import MicroDopplerDataset
import matplotlib.pyplot as plt

def analyze_blur_causes():
    """分析模糊原因"""
    print("🔍 VAE重建模糊问题分析")
    print("=" * 50)
    
    # 检查模型
    model_path = Path("/kaggle/working/outputs/vae_celeba_standard/final_model")
    if not model_path.exists():
        print("❌ 模型不存在，请先训练模型")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = AutoencoderKL.from_pretrained(str(model_path)).to(device).eval()
    
    # 加载数据
    dataset = MicroDopplerDataset(
        data_dir="/kaggle/input/dataset",
        resolution=64,
        augment=False,
        split="test"
    )
    
    print(f"📊 当前架构分析:")
    print(f"   输入: 64×64×3 = 12,288 像素")
    print(f"   潜在: 8×8×4 = 256 维度")
    print(f"   压缩比: {12288/256:.1f}:1")
    
    # 分析多个样本
    blur_scores = []
    detail_losses = []
    
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        original = sample['image'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            # 编码解码
            latent = vae.encode(original).latent_dist.sample()
            reconstructed = vae.decode(latent).sample
            
            # 计算细节损失 (高频信息)
            orig_np = original.cpu().squeeze().numpy()
            recon_np = reconstructed.cpu().squeeze().numpy()
            
            # 计算梯度幅度 (边缘信息)
            orig_grad = np.gradient(orig_np, axis=(1,2))
            recon_grad = np.gradient(recon_np, axis=(1,2))
            
            orig_edge = np.sqrt(orig_grad[0]**2 + orig_grad[1]**2).mean()
            recon_edge = np.sqrt(recon_grad[0]**2 + recon_grad[1]**2).mean()
            
            edge_preservation = recon_edge / (orig_edge + 1e-8)
            detail_losses.append(1 - edge_preservation)
            
            # 计算模糊度 (方差)
            orig_var = np.var(orig_np)
            recon_var = np.var(recon_np)
            blur_score = 1 - (recon_var / (orig_var + 1e-8))
            blur_scores.append(blur_score)
    
    avg_blur = np.mean(blur_scores)
    avg_detail_loss = np.mean(detail_losses)
    
    print(f"\n📈 模糊分析结果:")
    print(f"   平均模糊度: {avg_blur:.3f} (0=清晰, 1=完全模糊)")
    print(f"   边缘信息损失: {avg_detail_loss:.3f}")
    
    # 诊断和建议
    print(f"\n🔍 问题诊断:")
    
    if avg_blur > 0.3:
        print("⚠️  模糊度较高，主要原因:")
        print("   1. 压缩比过高 (48:1)")
        print("   2. 潜在空间维度不足")
        print("   3. 可能需要更复杂的解码器")
    
    if avg_detail_loss > 0.4:
        print("⚠️  细节损失严重，建议:")
        print("   1. 增加潜在空间维度")
        print("   2. 降低压缩比")
        print("   3. 使用更强的感知损失")
    
    print(f"\n💡 改进建议 (按优先级):")
    
    print(f"\n🎯 方案1: 降低压缩比")
    print(f"   当前: 64×64 → 8×8 (8倍下采样)")
    print(f"   建议: 64×64 → 16×16 (4倍下采样)")
    print(f"   效果: 压缩比 48:1 → 12:1")
    print(f"   优势: 保留更多细节信息")
    
    print(f"\n🎯 方案2: 增加潜在维度")
    print(f"   当前: 8×8×4 = 256维")
    print(f"   建议: 8×8×8 = 512维")
    print(f"   效果: 压缩比 48:1 → 24:1")
    print(f"   优势: 更大的信息容量")
    
    print(f"\n🎯 方案3: 增强感知损失")
    print(f"   当前: 感知损失权重 0.1")
    print(f"   建议: 感知损失权重 0.5-1.0")
    print(f"   效果: 更好的视觉质量")
    print(f"   注意: 需要确保LPIPS在GPU上运行")
    
    print(f"\n🎯 方案4: 使用更复杂的架构")
    print(f"   当前: 每层1个ResNet块")
    print(f"   建议: 每层2个ResNet块")
    print(f"   效果: 更强的特征提取能力")
    print(f"   代价: 增加计算量")
    
    # 推荐最佳方案
    print(f"\n⭐ 推荐方案 (平衡效果和计算量):")
    print(f"   1. 降低压缩比: 64×64 → 16×16")
    print(f"   2. 增加感知损失权重到 0.5")
    print(f"   3. 保持其他参数不变")
    print(f"   预期PSNR提升: 21.78 dB → 25+ dB")

def suggest_implementation():
    """建议具体实现"""
    print(f"\n🛠️  实现步骤:")
    print(f"   1. 修改架构: 减少下采样层")
    print(f"   2. 调整感知损失权重")
    print(f"   3. 重新训练并对比效果")
    print(f"   4. 如果效果好，可以进一步优化")

if __name__ == "__main__":
    analyze_blur_causes()
    suggest_implementation()
