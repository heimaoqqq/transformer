#!/usr/bin/env python3
"""
训练性能分析器 - 找出真正的瓶颈
"""

import os
import sys
import torch
import time
import psutil
from pathlib import Path

# 添加项目路径
sys.path.insert(0, '/kaggle/working/VAE')

def profile_model_creation():
    """分析模型创建时间"""
    print("🔍 分析模型创建...")
    
    from diffusers import AutoencoderKL
    
    # 3层下采样配置
    start_time = time.time()
    vae_3layer = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=[
            "DownEncoderBlock2D",  # 256→128
            "DownEncoderBlock2D",  # 128→64
            "DownEncoderBlock2D"   # 64→32
        ],
        up_block_types=[
            "UpDecoderBlock2D",    # 32→64
            "UpDecoderBlock2D",    # 64→128
            "UpDecoderBlock2D"     # 128→256
        ],
        block_out_channels=[128, 256, 512],
        latent_channels=4,
        sample_size=256,
        layers_per_block=2,
    )
    creation_time = time.time() - start_time
    
    # 计算参数量
    total_params = sum(p.numel() for p in vae_3layer.parameters())
    
    print(f"   ✅ 3层VAE创建时间: {creation_time:.2f}s")
    print(f"   📊 参数量: {total_params:,}")
    print(f"   💾 模型大小: {total_params * 4 / 1024**2:.1f} MB")
    
    # 对比4层配置
    start_time = time.time()
    vae_4layer = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=[
            "DownEncoderBlock2D",
            "DownEncoderBlock2D", 
            "DownEncoderBlock2D",
            "DownEncoderBlock2D"
        ],
        up_block_types=[
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D"
        ],
        block_out_channels=[128, 256, 512, 512],
        latent_channels=4,
        sample_size=256,
        layers_per_block=2,
    )
    creation_time_4 = time.time() - start_time
    
    total_params_4 = sum(p.numel() for p in vae_4layer.parameters())
    
    print(f"\n   📊 对比4层VAE:")
    print(f"   ⏱️  创建时间: {creation_time_4:.2f}s")
    print(f"   📊 参数量: {total_params_4:,}")
    print(f"   📉 参数减少: {(total_params_4 - total_params) / total_params_4 * 100:.1f}%")
    
    return vae_3layer

def profile_data_loading():
    """分析数据加载性能"""
    print("\n🔍 分析数据加载...")
    
    try:
        from utils.data_loader import MicroDopplerDataset
        from torch.utils.data import DataLoader
        
        # 创建数据集
        dataset = MicroDopplerDataset(
            data_dir="/kaggle/input/dataset",
            resolution=256,
            split="train"
        )
        
        print(f"   📊 数据集大小: {len(dataset)}")
        
        # 测试不同配置的数据加载器
        configs = [
            {"batch_size": 2, "num_workers": 0, "pin_memory": False},
            {"batch_size": 4, "num_workers": 0, "pin_memory": False},
            {"batch_size": 4, "num_workers": 1, "pin_memory": True},
            {"batch_size": 4, "num_workers": 2, "pin_memory": True},
        ]
        
        for config in configs:
            dataloader = DataLoader(dataset, **config)
            
            # 测试加载时间
            start_time = time.time()
            for i, batch in enumerate(dataloader):
                if i >= 5:  # 只测试前5个批次
                    break
            load_time = time.time() - start_time
            
            print(f"   ⏱️  配置 {config}: {load_time/5:.2f}s/批次")
            
    except Exception as e:
        print(f"   ❌ 数据加载分析失败: {e}")

def profile_forward_pass():
    """分析前向传播性能"""
    print("\n🔍 分析前向传播...")
    
    device = torch.device("cuda:0")
    vae = profile_model_creation()
    vae = vae.to(device)
    vae.eval()
    
    # 测试不同批次大小
    batch_sizes = [1, 2, 4, 6]
    
    for batch_size in batch_sizes:
        print(f"\n   📦 批次大小: {batch_size}")
        
        # 创建测试数据
        test_input = torch.randn(batch_size, 3, 256, 256).to(device)
        
        # 预热
        with torch.no_grad():
            _ = vae.encode(test_input).latent_dist.sample()
            _ = vae.decode(_).sample
        
        torch.cuda.synchronize()
        
        # 测试完整前向传播
        times = []
        for _ in range(10):
            torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.no_grad():
                posterior = vae.encode(test_input).latent_dist
                latents = posterior.sample()
                reconstruction = vae.decode(latents).sample
            
            torch.cuda.synchronize()
            times.append(time.time() - start_time)
        
        avg_time = sum(times) / len(times)
        throughput = batch_size / avg_time
        
        print(f"      ⏱️  前向传播: {avg_time*1000:.1f}ms")
        print(f"      🚀 吞吐量: {throughput:.1f} 样本/秒")
        
        # 内存使用
        memory_used = torch.cuda.max_memory_allocated() / 1024**3
        print(f"      💾 GPU内存: {memory_used:.1f} GB")
        
        torch.cuda.empty_cache()

def estimate_training_time():
    """估算训练时间"""
    print("\n🔍 估算训练时间...")
    
    try:
        from utils.data_loader import MicroDopplerDataset
        
        dataset = MicroDopplerDataset(
            data_dir="/kaggle/input/dataset",
            resolution=256,
            split="train"
        )
        
        dataset_size = len(dataset)
        batch_size = 4  # train_safe.py的配置
        steps_per_epoch = dataset_size // batch_size
        
        print(f"   📊 数据集: {dataset_size} 样本")
        print(f"   📦 批次大小: {batch_size}")
        print(f"   📈 每轮步数: {steps_per_epoch}")
        
        # 基于实际测试的时间估算
        estimated_time_per_step = 2.5  # 秒 (3层下采样应该更快)
        time_per_epoch = steps_per_epoch * estimated_time_per_step
        
        print(f"\n   ⏱️  时间估算:")
        print(f"      每步时间: ~{estimated_time_per_step:.1f}s")
        print(f"      每轮时间: ~{time_per_epoch/60:.1f}分钟")
        print(f"      30轮总时间: ~{time_per_epoch*30/3600:.1f}小时")
        
        # 如果还是30分钟一轮，说明有其他瓶颈
        if time_per_epoch > 25 * 60:  # 超过25分钟
            print(f"\n   ⚠️  警告: 预估时间仍然很长!")
            print(f"      可能的瓶颈:")
            print(f"      - 数据加载慢 (磁盘I/O)")
            print(f"      - 网络通信慢 (多GPU同步)")
            print(f"      - 内存碎片化")
            print(f"      - CPU瓶颈")
        
    except Exception as e:
        print(f"   ❌ 时间估算失败: {e}")

def check_system_resources():
    """检查系统资源"""
    print("\n🔍 检查系统资源...")
    
    # CPU信息
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()
    
    # 内存信息
    memory = psutil.virtual_memory()
    
    # GPU信息
    gpu_count = torch.cuda.device_count()
    
    print(f"   🖥️  CPU: {cpu_count}核, 使用率: {cpu_percent:.1f}%")
    print(f"   💾 RAM: {memory.total/1024**3:.1f}GB, 使用率: {memory.percent:.1f}%")
    print(f"   🎮 GPU: {gpu_count}个")
    
    if torch.cuda.is_available():
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_used = torch.cuda.memory_allocated(i) / 1024**3
            memory_total = props.total_memory / 1024**3
            print(f"      GPU {i}: {props.name}, {memory_used:.1f}/{memory_total:.1f}GB")

def main():
    """主函数"""
    print("🔍 VAE训练性能分析")
    print("=" * 80)
    
    # 检查系统资源
    check_system_resources()
    
    # 分析模型
    profile_model_creation()
    
    # 分析数据加载
    profile_data_loading()
    
    # 分析前向传播
    profile_forward_pass()
    
    # 估算训练时间
    estimate_training_time()
    
    print("\n" + "=" * 80)
    print("✅ 性能分析完成!")
    print("\n💡 如果训练仍然很慢，可能的原因:")
    print("   1. 数据加载瓶颈 (磁盘I/O)")
    print("   2. 多GPU通信开销")
    print("   3. 梯度累积延迟")
    print("   4. 系统资源竞争")

if __name__ == "__main__":
    main()
