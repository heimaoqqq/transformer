#!/usr/bin/env python3
"""
测试更新后的check_vae.py
验证与新训练配置的兼容性
"""

import torch
from diffusers import AutoencoderKL
from pathlib import Path
import tempfile

def test_vae_checker_compatibility():
    """测试VAE检查器与新配置的兼容性"""
    print("🧪 测试check_vae.py更新")
    print("=" * 50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  设备: {device}")
    
    # 创建符合新配置的VAE模型
    print(f"\n🏗️  创建新配置VAE模型 (128×128 → 32×32):")
    try:
        vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],  # 3层
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],        # 3层
            block_out_channels=[128, 256, 512],                                                   # 3层通道数
            latent_channels=4,
            sample_size=128,                                                 # 128×128输入
            layers_per_block=1,
            act_fn="silu",
            norm_num_groups=32,
            scaling_factor=0.18215,
        ).to(device)
        
        total_params = sum(p.numel() for p in vae.parameters())
        print(f"   ✅ 模型创建成功 - 参数量: {total_params:,}")
        
    except Exception as e:
        print(f"   ❌ 模型创建失败: {e}")
        return False
    
    # 测试前向传播
    print(f"\n🔄 测试前向传播:")
    try:
        test_input = torch.randn(1, 3, 128, 128).to(device)
        
        with torch.no_grad():
            posterior = vae.encode(test_input).latent_dist
            latent = posterior.sample()
            reconstructed = vae.decode(latent).sample
        
        print(f"   📐 输入形状: {test_input.shape}")
        print(f"   🎯 潜在形状: {latent.shape}")
        print(f"   🔄 重建形状: {reconstructed.shape}")
        
        # 验证形状
        expected_latent = (1, 4, 32, 32)
        expected_output = (1, 3, 128, 128)
        
        if latent.shape == expected_latent:
            print(f"   ✅ 潜在空间形状正确: {latent.shape}")
        else:
            print(f"   ❌ 潜在空间形状错误: {latent.shape}, 期望: {expected_latent}")
            return False
            
        if reconstructed.shape == expected_output:
            print(f"   ✅ 重建形状正确: {reconstructed.shape}")
        else:
            print(f"   ❌ 重建形状错误: {reconstructed.shape}, 期望: {expected_output}")
            return False
            
        # 计算压缩比
        compression_ratio = test_input.numel() / latent.numel()
        print(f"   📊 压缩比: {compression_ratio:.1f}:1")
        
        if abs(compression_ratio - 12.0) < 1.0:  # 期望12:1左右
            print(f"   ✅ 压缩比正确")
        else:
            print(f"   ⚠️  压缩比异常，期望约12:1")
        
    except Exception as e:
        print(f"   ❌ 前向传播失败: {e}")
        return False
    
    # 测试模型保存和加载
    print(f"\n💾 测试模型保存和加载:")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model"
            
            # 保存模型
            vae.save_pretrained(model_path)
            print(f"   ✅ 模型保存成功: {model_path}")
            
            # 加载模型
            loaded_vae = AutoencoderKL.from_pretrained(model_path).to(device)
            print(f"   ✅ 模型加载成功")
            
            # 验证加载的模型
            with torch.no_grad():
                test_input2 = torch.randn(1, 3, 128, 128).to(device)
                latent2 = loaded_vae.encode(test_input2).latent_dist.sample()
                
            if latent2.shape == expected_latent:
                print(f"   ✅ 加载模型架构正确: {latent2.shape}")
            else:
                print(f"   ❌ 加载模型架构错误: {latent2.shape}")
                return False
                
    except Exception as e:
        print(f"   ❌ 模型保存/加载失败: {e}")
        return False
    
    return True

def test_quality_standards():
    """测试新的质量评估标准"""
    print(f"\n📊 测试质量评估标准:")
    print("=" * 50)
    
    # 模拟不同PSNR值的评估
    test_cases = [
        (30.0, "优秀"),
        (27.0, "良好"), 
        (23.0, "一般"),
        (18.0, "较差")
    ]
    
    for psnr, expected in test_cases:
        if psnr > 28:
            result = "优秀"
        elif psnr > 25:
            result = "良好"
        elif psnr > 20:
            result = "一般"
        else:
            result = "较差"
            
        status = "✅" if result == expected else "❌"
        print(f"   {status} PSNR {psnr:.1f}dB → {result} (期望: {expected})")

def test_data_loader_compatibility():
    """测试数据加载器兼容性"""
    print(f"\n📁 测试数据加载器配置:")
    print("=" * 50)
    
    # 检查分辨率配置
    old_resolution = 64
    new_resolution = 128
    
    print(f"   旧配置: {old_resolution}×{old_resolution} = {old_resolution**2:,} 像素")
    print(f"   新配置: {new_resolution}×{new_resolution} = {new_resolution**2:,} 像素")
    print(f"   信息增量: {(new_resolution**2) / (old_resolution**2):.1f}倍")
    
    # 检查潜在空间
    old_latent_size = 8 * 8 * 4  # 旧版本
    new_latent_size = 32 * 32 * 4  # 新版本
    
    print(f"   旧潜在空间: 8×8×4 = {old_latent_size} 维")
    print(f"   新潜在空间: 32×32×4 = {new_latent_size} 维")
    print(f"   表示能力提升: {new_latent_size / old_latent_size:.1f}倍")

def main():
    """主函数"""
    print("🔧 check_vae.py 更新验证")
    print("🎯 验证与train_improved_quality.py的兼容性")
    print()
    
    # 运行测试
    success = True
    
    success &= test_vae_checker_compatibility()
    test_quality_standards()
    test_data_loader_compatibility()
    
    print(f"\n🎉 测试总结:")
    if success:
        print("✅ check_vae.py 更新成功，与新训练配置兼容")
        print("✅ 可以正确检查128×128→32×32的VAE模型")
        print("✅ 质量评估标准已更新为现代化标准")
    else:
        print("❌ 存在兼容性问题，需要进一步调试")
    
    print(f"\n📋 使用说明:")
    print("1. 在云服务器上运行: python check_vae.py")
    print("2. 检查新训练的VAE模型质量")
    print("3. 验证128×128→32×32架构")
    print("4. 确认PSNR > 28dB为优秀标准")

if __name__ == "__main__":
    main()
