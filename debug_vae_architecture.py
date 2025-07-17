#!/usr/bin/env python3
"""
调试VAE架构 - 验证下采样行为
"""

import torch
from diffusers import AutoencoderKL

def test_vae_downsampling():
    """测试不同配置的VAE下采样行为"""
    print("🔍 调试VAE架构下采样行为")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 测试配置1: 当前配置 (2层DownEncoderBlock2D)
    print("\n📊 测试配置1: 2层DownEncoderBlock2D")
    try:
        vae1 = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            block_out_channels=[128, 256],
            latent_channels=4,
            sample_size=128,
            layers_per_block=1,
            act_fn="silu",
            norm_num_groups=32,
            scaling_factor=0.18215,
        ).to(device)
        
        test_input = torch.randn(1, 3, 128, 128).to(device)
        with torch.no_grad():
            latent = vae1.encode(test_input).latent_dist.sample()
            reconstructed = vae1.decode(latent).sample
        
        print(f"   输入: {test_input.shape}")
        print(f"   潜在: {latent.shape}")
        print(f"   重建: {reconstructed.shape}")
        print(f"   下采样因子: {128 // latent.shape[-1]}")
        
    except Exception as e:
        print(f"   ❌ 配置1失败: {e}")
    
    # 测试配置2: 3层DownEncoderBlock2D (标准Stable Diffusion)
    print("\n📊 测试配置2: 3层DownEncoderBlock2D (标准SD)")
    try:
        vae2 = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
            block_out_channels=[128, 256, 512],
            latent_channels=4,
            sample_size=128,
            layers_per_block=1,
            act_fn="silu",
            norm_num_groups=32,
            scaling_factor=0.18215,
        ).to(device)
        
        test_input = torch.randn(1, 3, 128, 128).to(device)
        with torch.no_grad():
            latent = vae2.encode(test_input).latent_dist.sample()
            reconstructed = vae2.decode(latent).sample
        
        print(f"   输入: {test_input.shape}")
        print(f"   潜在: {latent.shape}")
        print(f"   重建: {reconstructed.shape}")
        print(f"   下采样因子: {128 // latent.shape[-1]}")
        
    except Exception as e:
        print(f"   ❌ 配置2失败: {e}")
    
    # 测试配置3: 4层DownEncoderBlock2D
    print("\n📊 测试配置3: 4层DownEncoderBlock2D")
    try:
        vae3 = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
            block_out_channels=[128, 256, 512, 512],
            latent_channels=4,
            sample_size=128,
            layers_per_block=1,
            act_fn="silu",
            norm_num_groups=32,
            scaling_factor=0.18215,
        ).to(device)
        
        test_input = torch.randn(1, 3, 128, 128).to(device)
        with torch.no_grad():
            latent = vae3.encode(test_input).latent_dist.sample()
            reconstructed = vae3.decode(latent).sample
        
        print(f"   输入: {test_input.shape}")
        print(f"   潜在: {latent.shape}")
        print(f"   重建: {reconstructed.shape}")
        print(f"   下采样因子: {128 // latent.shape[-1]}")
        
    except Exception as e:
        print(f"   ❌ 配置3失败: {e}")
    
    # 测试不同的layers_per_block
    print("\n📊 测试配置4: 2层DownEncoderBlock2D + layers_per_block=2")
    try:
        vae4 = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            block_out_channels=[128, 256],
            latent_channels=4,
            sample_size=128,
            layers_per_block=2,  # 增加每层的ResNet块数
            act_fn="silu",
            norm_num_groups=32,
            scaling_factor=0.18215,
        ).to(device)
        
        test_input = torch.randn(1, 3, 128, 128).to(device)
        with torch.no_grad():
            latent = vae4.encode(test_input).latent_dist.sample()
            reconstructed = vae4.decode(latent).sample
        
        print(f"   输入: {test_input.shape}")
        print(f"   潜在: {latent.shape}")
        print(f"   重建: {reconstructed.shape}")
        print(f"   下采样因子: {128 // latent.shape[-1]}")
        
    except Exception as e:
        print(f"   ❌ 配置4失败: {e}")
    
    print("\n🎯 结论:")
    print("   - 每个DownEncoderBlock2D进行1次下采样 (2倍)")
    print("   - 要达到128→32需要3层DownEncoderBlock2D (2^3=8倍)")
    print("   - 当前2层配置只能达到128→64 (2^2=4倍)")
    print("   - layers_per_block不影响下采样倍数，只影响特征提取能力")

if __name__ == "__main__":
    test_vae_downsampling()
