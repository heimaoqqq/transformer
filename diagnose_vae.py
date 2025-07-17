#!/usr/bin/env python3
"""
VAE训练问题诊断工具
分析损失组成和参数设置
"""

import torch
import numpy as np
from pathlib import Path
from diffusers import AutoencoderKL
from utils.data_loader import MicroDopplerDataset

def diagnose_vae_issues():
    """诊断VAE训练问题"""
    print("🔍 VAE训练问题诊断")
    print("=" * 50)
    
    # 1. 检查模型
    model_path = Path("/kaggle/working/outputs/vae_celeba_standard/final_model")
    if not model_path.exists():
        print("❌ 模型不存在")
        return
    
    print("✅ 加载模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = AutoencoderKL.from_pretrained(str(model_path)).to(device).eval()
    
    # 2. 检查数据
    print("\n📊 检查数据...")
    dataset = MicroDopplerDataset(
        data_dir="/kaggle/input/dataset",
        resolution=64,
        augment=False,
        split="test"
    )
    print(f"数据集大小: {len(dataset)}")
    
    # 3. 分析单个样本的重建
    print("\n🔬 分析重建过程...")
    sample = dataset[0]
    original = sample['image'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        # 编码
        posterior = vae.encode(original).latent_dist
        latent_mean = posterior.mean
        latent_std = posterior.std
        latent_sample = posterior.sample()
        
        # 解码
        reconstructed = vae.decode(latent_sample).sample
        
        print(f"原始图像范围: [{original.min():.3f}, {original.max():.3f}]")
        print(f"重建图像范围: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")
        print(f"潜在均值: {latent_mean.mean():.3f} ± {latent_mean.std():.3f}")
        print(f"潜在标准差: {latent_std.mean():.3f} ± {latent_std.std():.3f}")
        
        # 计算各种损失
        mse_loss = torch.nn.functional.mse_loss(reconstructed, original)
        
        # KL散度
        kl_loss = -0.5 * torch.sum(1 + torch.log(latent_std.pow(2)) - latent_mean.pow(2) - latent_std.pow(2))
        kl_loss = kl_loss / original.numel()
        
        print(f"\n📈 损失分析:")
        print(f"MSE损失: {mse_loss:.6f}")
        print(f"KL损失: {kl_loss:.6f}")
        print(f"KL损失 × 1e-4: {kl_loss * 1e-4:.6f}")
        print(f"KL损失 × 1e-6: {kl_loss * 1e-6:.6f}")
        
        # PSNR
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse_loss))
        print(f"PSNR: {psnr:.2f} dB")
        
    # 4. 检查感知损失
    print("\n🎯 检查感知损失...")
    try:
        import lpips
        lpips_loss = lpips.LPIPS(net='vgg').to(device)
        
        with torch.no_grad():
            # 归一化到[-1, 1]
            recon_norm = reconstructed * 2.0 - 1.0
            orig_norm = original * 2.0 - 1.0
            perceptual_loss = lpips_loss(recon_norm, orig_norm).mean()
            print(f"✅ 感知损失: {perceptual_loss:.6f}")
            
    except Exception as e:
        print(f"❌ 感知损失失败: {e}")
    
    # 5. 建议
    print("\n💡 诊断建议:")
    
    if kl_loss > 10:
        print("⚠️  KL损失过高，建议降低KL权重到1e-6")
    
    if mse_loss > 0.1:
        print("⚠️  MSE损失过高，可能需要:")
        print("   - 降低学习率到1e-4")
        print("   - 增加训练轮数")
        print("   - 检查数据预处理")
    
    if psnr < 15:
        print("⚠️  PSNR过低，建议:")
        print("   - 确保感知损失正常工作")
        print("   - 调整损失权重平衡")
        print("   - 检查模型架构是否合适")

def suggest_fixes():
    """建议修复方案"""
    print("\n🔧 修复建议:")
    print("1. 降低KL权重: 1e-4 → 1e-6")
    print("2. 降低学习率: 2e-4 → 1e-4") 
    print("3. 确保感知损失在GPU上运行")
    print("4. 增加训练轮数到100轮")
    print("5. 检查数据预处理是否正确")

if __name__ == "__main__":
    diagnose_vae_issues()
    suggest_fixes()
