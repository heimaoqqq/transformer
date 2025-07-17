#!/usr/bin/env python3
"""
快速测试VAE配置
验证新的损失权重是否合理
"""

import torch
import numpy as np
from diffusers import AutoencoderKL
from utils.data_loader import MicroDopplerDataset

def test_vae_config():
    """测试VAE配置"""
    print("🧪 快速测试VAE配置")
    print("=" * 40)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    
    # 创建一个小的VAE用于测试
    from diffusers import AutoencoderKL
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
        up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
        block_out_channels=[64, 128, 256],
        layers_per_block=1,
        latent_channels=4,
        sample_size=64,
    ).to(device)
    
    # 测试数据
    dataset = MicroDopplerDataset(
        data_dir="/kaggle/input/dataset",
        resolution=64,
        augment=False,
        split="test"
    )
    
    sample = dataset[0]
    test_image = sample['image'].unsqueeze(0).to(device)
    
    print(f"测试图像形状: {test_image.shape}")
    print(f"测试图像范围: [{test_image.min():.3f}, {test_image.max():.3f}]")
    
    # 前向传播
    with torch.no_grad():
        posterior = vae.encode(test_image).latent_dist
        latent = posterior.sample()
        reconstructed = vae.decode(latent).sample
        
        # 计算损失
        mse_loss = torch.nn.functional.mse_loss(reconstructed, test_image)
        
        # KL散度
        kl_loss = -0.5 * torch.sum(1 + torch.log(posterior.std.pow(2)) - posterior.mean.pow(2) - posterior.std.pow(2))
        kl_loss = kl_loss / test_image.numel()
        
        print(f"\n📊 损失分析:")
        print(f"MSE损失: {mse_loss:.6f}")
        print(f"KL损失: {kl_loss:.6f}")
        
        # 不同KL权重的影响
        print(f"\n⚖️  不同KL权重的总损失:")
        print(f"KL权重 1e-4: {mse_loss + kl_loss * 1e-4:.6f}")
        print(f"KL权重 1e-5: {mse_loss + kl_loss * 1e-5:.6f}")
        print(f"KL权重 1e-6: {mse_loss + kl_loss * 1e-6:.6f}")
        
        # PSNR
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse_loss))
        print(f"\nPSNR: {psnr:.2f} dB")
        
        # 建议
        print(f"\n💡 建议:")
        if kl_loss > 10:
            print("✅ KL损失较高，使用1e-6权重是正确的")
        else:
            print("⚠️  KL损失较低，可以考虑稍高的权重")
            
        if mse_loss > 0.1:
            print("⚠️  MSE损失较高，需要更多训练")
        else:
            print("✅ MSE损失合理")

if __name__ == "__main__":
    test_vae_config()
