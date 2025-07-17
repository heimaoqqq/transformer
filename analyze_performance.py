#!/usr/bin/env python3
"""
VAE训练性能分析工具
"""

import torch
import time
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, '/kaggle/working/VAE')

def analyze_vae_architecture():
    """分析VAE架构"""
    print("🔍 VAE架构分析")
    print("=" * 50)
    
    from diffusers import AutoencoderKL
    
    # 当前配置
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=[
            "DownEncoderBlock2D",
            "DownEncoderBlock2D", 
            "DownEncoderBlock2D",
            "DownEncoderBlock2D"  # 4层下采样
        ],
        up_block_types=[
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D"    # 4层上采样
        ],
        block_out_channels=[128, 256, 512, 512],
        latent_channels=4,
        sample_size=256,
        layers_per_block=2,
    )
    
    # 计算参数量
    total_params = sum(p.numel() for p in vae.parameters())
    trainable_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    
    print(f"📊 模型参数:")
    print(f"   总参数量: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")
    print(f"   模型大小: {total_params * 4 / 1024**2:.1f} MB (FP32)")
    
    # 分析压缩比
    input_size = 256 * 256 * 3  # 输入像素数
    latent_size = 32 * 32 * 4   # 潜在空间大小 (256/8 = 32)
    compression_ratio = input_size / latent_size
    
    print(f"\n📐 压缩分析:")
    print(f"   输入尺寸: 256×256×3 = {input_size:,} 像素")
    print(f"   潜在尺寸: 32×32×4 = {latent_size:,} 像素")
    print(f"   压缩比: {compression_ratio:.1f}:1")
    print(f"   下采样倍数: 2^4 = 16倍 (每边)")
    
    return vae

def benchmark_forward_pass():
    """基准测试前向传播"""
    print("\n🚀 前向传播基准测试")
    print("=" * 50)
    
    device = torch.device("cuda:0")
    vae = analyze_vae_architecture()
    vae = vae.to(device)
    vae.eval()
    
    # 测试不同批次大小
    batch_sizes = [1, 2, 4, 8]
    
    for batch_size in batch_sizes:
        print(f"\n📊 批次大小: {batch_size}")
        
        # 创建测试数据
        test_input = torch.randn(batch_size, 3, 256, 256).to(device)
        
        # 预热
        with torch.no_grad():
            _ = vae.encode(test_input).latent_dist.sample()
        
        torch.cuda.synchronize()
        
        # 测试编码
        times = []
        for _ in range(10):
            torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.no_grad():
                posterior = vae.encode(test_input).latent_dist
                latents = posterior.sample()
            
            torch.cuda.synchronize()
            times.append(time.time() - start_time)
        
        encode_time = sum(times) / len(times)
        
        # 测试解码
        times = []
        for _ in range(10):
            torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.no_grad():
                reconstruction = vae.decode(latents).sample
            
            torch.cuda.synchronize()
            times.append(time.time() - start_time)
        
        decode_time = sum(times) / len(times)
        
        total_time = encode_time + decode_time
        throughput = batch_size / total_time
        
        print(f"   编码时间: {encode_time*1000:.1f}ms")
        print(f"   解码时间: {decode_time*1000:.1f}ms")
        print(f"   总时间: {total_time*1000:.1f}ms")
        print(f"   吞吐量: {throughput:.1f} 样本/秒")
        
        # 内存使用
        memory_used = torch.cuda.max_memory_allocated() / 1024**3
        print(f"   GPU内存: {memory_used:.1f} GB")
        
        torch.cuda.empty_cache()

def analyze_dataset_size():
    """分析数据集大小"""
    print("\n📁 数据集分析")
    print("=" * 50)
    
    try:
        from utils.data_loader import MicroDopplerDataset
        
        dataset = MicroDopplerDataset(
            data_dir="/kaggle/input/dataset",
            resolution=256,
            split="train"
        )
        
        dataset_size = len(dataset)
        batch_size = 4  # 当前配置
        steps_per_epoch = dataset_size // batch_size
        
        print(f"📊 数据集信息:")
        print(f"   数据集大小: {dataset_size:,} 样本")
        print(f"   批次大小: {batch_size}")
        print(f"   每轮步数: {steps_per_epoch:,}")
        
        # 估算训练时间
        estimated_time_per_step = 4.0  # 秒 (基于观察)
        time_per_epoch = steps_per_epoch * estimated_time_per_step
        
        print(f"\n⏱️  时间估算:")
        print(f"   每步时间: ~{estimated_time_per_step:.1f}秒")
        print(f"   每轮时间: ~{time_per_epoch/60:.1f}分钟")
        print(f"   40轮总时间: ~{time_per_epoch*40/3600:.1f}小时")
        
        return dataset_size, steps_per_epoch
        
    except Exception as e:
        print(f"❌ 数据集分析失败: {e}")
        return None, None

def suggest_optimizations():
    """建议优化方案"""
    print("\n💡 性能优化建议")
    print("=" * 50)
    
    print("🚀 速度优化方案:")
    print("   1. 增加批次大小 (如果内存允许)")
    print("      - 当前: batch_size=4")
    print("      - 建议: batch_size=6-8")
    print("      - 效果: 提升30-50%吞吐量")
    
    print("\n   2. 减少模型复杂度")
    print("      - 当前: 4层下采样 + 2层/块")
    print("      - 建议: 3层下采样 或 1层/块")
    print("      - 效果: 减少50%计算量")
    
    print("\n   3. 优化数据加载")
    print("      - 当前: num_workers=1")
    print("      - 建议: num_workers=2-4")
    print("      - 效果: 减少数据等待时间")
    
    print("\n   4. 使用更高效的优化器")
    print("      - 当前: AdamW")
    print("      - 建议: AdamW + 学习率调度")
    print("      - 效果: 更快收敛")
    
    print("\n🎯 质量vs速度权衡:")
    print("   - 高质量: 256×256, 4层, batch_size=4 (当前)")
    print("   - 平衡: 256×256, 3层, batch_size=6")
    print("   - 快速: 128×128, 3层, batch_size=8")

def main():
    """主函数"""
    print("🔍 VAE训练性能分析")
    print("=" * 80)
    
    # 分析架构
    vae = analyze_vae_architecture()
    
    # 基准测试
    benchmark_forward_pass()
    
    # 分析数据集
    analyze_dataset_size()
    
    # 优化建议
    suggest_optimizations()
    
    print("\n" + "=" * 80)
    print("✅ 性能分析完成!")

if __name__ == "__main__":
    main()
