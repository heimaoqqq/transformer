#!/usr/bin/env python3
"""
测试128×128 → 32×32配置
验证架构、显存使用和数据加载
"""

import torch
import numpy as np
from pathlib import Path
from diffusers import AutoencoderKL
from utils.data_loader import MicroDopplerDataset
from torch.utils.data import DataLoader
import time

def test_new_architecture():
    """测试新架构配置"""
    print("🧪 测试128×128 → 32×32现代化配置")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not torch.cuda.is_available():
        print("❌ 需要CUDA支持进行测试")
        return False
    
    gpu_props = torch.cuda.get_device_properties(0)
    gpu_memory = gpu_props.total_memory / 1024**3
    print(f"🎮 GPU: {gpu_props.name} ({gpu_memory:.1f}GB)")
    
    # 测试数据加载器
    print(f"\n📊 测试数据加载器 (128×128 + Lanczos):")
    try:
        dataset = MicroDopplerDataset(
            data_dir="/kaggle/input/dataset",
            resolution=128,  # 新的分辨率
            augment=False,
            split="test"
        )
        
        # 测试单个样本
        sample = dataset[0]
        print(f"   ✅ 样本形状: {sample['image'].shape}")
        print(f"   ✅ 数据类型: {sample['image'].dtype}")
        print(f"   ✅ 数值范围: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
        
    except Exception as e:
        print(f"   ❌ 数据加载失败: {e}")
        return False
    
    # 测试VAE架构
    print(f"\n🏗️  测试VAE架构 (128×128 → 32×32):")
    try:
        # 新架构配置 (2层下采样: 128→64→32)
        vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],  # 2层
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],        # 2层
            block_out_channels=[128, 256],                                   # 2层通道数
            latent_channels=4,
            sample_size=128,                                                 # 修复: 设置为128匹配输入尺寸
            layers_per_block=1,                                              # 标准配置
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
            else:
                print(f"   ❌ 潜在空间形状错误: {latent.shape}, 期望: {expected_latent_shape}")
                return False
                
            if reconstructed.shape == expected_output_shape:
                print(f"   ✅ 重建形状正确: {reconstructed.shape}")
            else:
                print(f"   ❌ 重建形状错误: {reconstructed.shape}, 期望: {expected_output_shape}")
                return False
        
    except Exception as e:
        print(f"   ❌ VAE架构测试失败: {e}")
        return False
    
    # 测试显存使用
    print(f"\n💾 测试显存使用:")
    try:
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() / 1024**2
        
        # 测试不同批次大小
        batch_sizes = [2, 4, 6, 8]
        max_batch = 0
        
        for batch_size in batch_sizes:
            try:
                torch.cuda.empty_cache()
                test_batch = torch.randn(batch_size, 3, 128, 128).to(device)
                
                with torch.no_grad():
                    latent = vae.encode(test_batch).latent_dist.sample()
                    reconstructed = vae.decode(latent).sample
                
                current_memory = torch.cuda.memory_allocated() / 1024**2
                memory_used = current_memory - initial_memory
                
                print(f"   ✅ 批次{batch_size}: {memory_used:.0f}MB")
                max_batch = batch_size
                
                del test_batch, latent, reconstructed
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"   ⚠️  批次{batch_size}: 显存不足")
                    break
                else:
                    raise e
        
        print(f"   📊 推荐最大批次: {max_batch}")
        
    except Exception as e:
        print(f"   ❌ 显存测试失败: {e}")
        return False
    
    # 测试数据加载器性能
    print(f"\n⏱️  测试数据加载性能:")
    try:
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        start_time = time.time()
        for i, batch in enumerate(dataloader):
            if i >= 5:  # 只测试5个批次
                break
        
        elapsed = time.time() - start_time
        print(f"   ✅ 5个批次加载时间: {elapsed:.2f}秒")
        print(f"   ✅ 平均每批次: {elapsed/5:.2f}秒")
        
    except Exception as e:
        print(f"   ❌ 数据加载性能测试失败: {e}")
        return False
    
    # 计算改进指标
    print(f"\n📈 配置对比:")
    print(f"   旧配置 (64×64 → 8×8):")
    print(f"     - 输入: 64×64×3 = 12,288 像素")
    print(f"     - 潜在: 8×8×4 = 256 维度")
    print(f"     - 压缩比: 48:1")
    print(f"     - 通道: [64, 128, 256]")
    
    print(f"   新配置 (128×128 → 32×32):")
    print(f"     - 输入: 128×128×3 = 49,152 像素")
    print(f"     - 潜在: 32×32×4 = 4,096 维度")
    print(f"     - 压缩比: 12:1")
    print(f"     - 通道: [128, 256]")
    
    print(f"   改进:")
    print(f"     - 输入分辨率: 4倍提升")
    print(f"     - 信息容量: 16倍提升")
    print(f"     - 压缩比: 4倍降低 (更好)")
    print(f"     - 缩放质量: Lanczos (最佳)")
    print(f"     - 关键修复: sample_size=128 (匹配输入尺寸，确保正确下采样)")
    
    return True

def main():
    """主函数"""
    print("🎯 128×128 → 32×32 现代化配置测试")
    
    success = test_new_architecture()
    
    if success:
        print(f"\n🎉 配置测试通过!")
        print(f"✅ 可以开始训练: python train_improved_quality.py")
        print(f"📊 预期PSNR: 28+ dB (vs 之前21.78 dB)")
    else:
        print(f"\n❌ 配置测试失败!")
        print(f"🔧 请检查GPU内存和数据路径")

if __name__ == "__main__":
    main()
