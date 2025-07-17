#!/usr/bin/env python3
"""
CPU版本VAE架构测试
"""

import torch
from diffusers import AutoencoderKL

def test_vae_architecture_cpu():
    """在CPU上测试VAE架构"""
    print("🧪 CPU版本VAE架构测试")
    print("=" * 50)
    
    device = "cpu"
    
    # 测试修复后的配置 (3层下采样)
    print("\n🏗️  测试修复后的配置 (3层DownEncoderBlock2D):")
    try:
        vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],  # 3层
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],        # 3层
            block_out_channels=[128, 256, 512],                                                   # 3层通道数
            latent_channels=4,
            sample_size=128,
            layers_per_block=1,
            act_fn="silu",
            norm_num_groups=32,
            scaling_factor=0.18215,
        ).to(device)
        
        # 计算参数量
        total_params = sum(p.numel() for p in vae.parameters())
        trainable_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
        
        print(f"   ✅ 总参数: {total_params:,}")
        print(f"   ✅ 可训练参数: {trainable_params:,}")
        
        # 测试前向传播
        test_input = torch.randn(1, 3, 128, 128).to(device)
        
        with torch.no_grad():
            # 编码
            latent_dist = vae.encode(test_input).latent_dist
            latent = latent_dist.sample()
            print(f"   ✅ 潜在空间形状: {latent.shape}")
            
            # 解码
            reconstructed = vae.decode(latent).sample
            print(f"   ✅ 重建形状: {reconstructed.shape}")
            
            # 验证形状
            expected_latent_shape = (1, 4, 32, 32)
            expected_output_shape = (1, 3, 128, 128)
            
            if latent.shape == expected_latent_shape:
                print(f"   ✅ 潜在空间形状正确: {latent.shape}")
                success = True
            else:
                print(f"   ❌ 潜在空间形状错误: {latent.shape}, 期望: {expected_latent_shape}")
                success = False
                
            if reconstructed.shape == expected_output_shape:
                print(f"   ✅ 重建形状正确: {reconstructed.shape}")
            else:
                print(f"   ❌ 重建形状错误: {reconstructed.shape}, 期望: {expected_output_shape}")
                success = False
        
        # 计算压缩比
        input_pixels = 128 * 128 * 3
        latent_pixels = 32 * 32 * 4
        compression_ratio = input_pixels / latent_pixels
        
        print(f"   📊 压缩比: {compression_ratio:.1f}:1")
        print(f"   📐 输入像素: {input_pixels:,}")
        print(f"   🎯 潜在像素: {latent_pixels:,}")
        
        return success
        
    except Exception as e:
        print(f"   ❌ VAE架构测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🎯 VAE架构修复验证")
    
    success = test_vae_architecture_cpu()
    
    if success:
        print(f"\n🎉 架构修复成功!")
        print(f"✅ 现在可以正确实现 128×128 → 32×32 下采样")
        print(f"✅ 压缩比: 12:1 (符合预期)")
        print(f"✅ 可以开始训练: python train_improved_quality.py")
    else:
        print(f"\n❌ 架构仍有问题!")
        print(f"🔧 需要进一步调试")

if __name__ == "__main__":
    main()
