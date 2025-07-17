#!/usr/bin/env python3
"""
调试版训练脚本 - 找出进度条不动的原因
"""

import os
import sys
import subprocess
import torch
import time
from pathlib import Path

def setup_debug_environment():
    """设置调试环境"""
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
    os.environ['PYTHONUNBUFFERED'] = '1'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 启用同步执行便于调试

def test_data_loading():
    """测试数据加载"""
    print("🔍 测试数据加载...")
    
    try:
        sys.path.insert(0, '/kaggle/working/VAE')
        from utils.data_loader import MicroDopplerDataset
        from torch.utils.data import DataLoader
        
        print("   ✅ 导入成功")
        
        # 创建数据集
        dataset = MicroDopplerDataset(
            data_dir="/kaggle/input/dataset",
            resolution=256,
            split="train"
        )
        
        print(f"   ✅ 数据集大小: {len(dataset)}")
        
        # 创建数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=2,  # 小批次测试
            shuffle=False,
            num_workers=0,  # 单线程
            pin_memory=True
        )
        
        print(f"   ✅ 数据加载器创建成功，批次数: {len(dataloader)}")
        
        # 测试加载第一个批次
        print("   🔄 测试加载第一个批次...")
        start_time = time.time()
        
        for i, batch in enumerate(dataloader):
            load_time = time.time() - start_time
            print(f"   ✅ 批次 {i} 加载成功 ({load_time:.2f}s)")
            print(f"      图像形状: {batch['image'].shape}")
            print(f"      图像范围: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")
            
            if i >= 2:  # 只测试前3个批次
                break
            
            start_time = time.time()
        
        return True
        
    except Exception as e:
        print(f"   ❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_forward():
    """测试模型前向传播"""
    print("\n🔍 测试模型前向传播...")
    
    try:
        from diffusers import AutoencoderKL
        
        print("   🔄 创建VAE模型...")
        vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            down_block_types=[
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
            ],
            up_block_types=[
                "UpDecoderBlock2D",
                "UpDecoderBlock2D", 
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
            ],
            block_out_channels=[128, 256, 512, 512],
            latent_channels=4,
            sample_size=256,
        )
        
        print("   ✅ VAE模型创建成功")
        
        # 移动到GPU
        device = torch.device("cuda:0")
        vae = vae.to(device)
        print(f"   ✅ 模型移动到 {device}")
        
        # 创建测试数据
        batch_size = 2
        test_images = torch.randn(batch_size, 3, 256, 256).to(device)
        print(f"   ✅ 测试数据创建: {test_images.shape}")
        
        # 测试编码
        print("   🔄 测试VAE编码...")
        start_time = time.time()
        
        with torch.no_grad():
            posterior = vae.encode(test_images).latent_dist
            latents = posterior.sample()
        
        encode_time = time.time() - start_time
        print(f"   ✅ 编码成功 ({encode_time:.2f}s): {latents.shape}")
        
        # 测试解码
        print("   🔄 测试VAE解码...")
        start_time = time.time()
        
        with torch.no_grad():
            reconstruction = vae.decode(latents).sample
        
        decode_time = time.time() - start_time
        print(f"   ✅ 解码成功 ({decode_time:.2f}s): {reconstruction.shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_accelerate():
    """测试Accelerate"""
    print("\n🔍 测试Accelerate...")
    
    try:
        from accelerate import Accelerator
        
        accelerator = Accelerator(
            gradient_accumulation_steps=4,
            mixed_precision="fp16"
        )
        
        print(f"   ✅ Accelerator创建成功")
        print(f"      设备: {accelerator.device}")
        print(f"      进程数: {accelerator.num_processes}")
        print(f"      分布式类型: {accelerator.distributed_type}")
        print(f"      是否主进程: {accelerator.is_main_process}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Accelerate测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def launch_debug_training():
    """启动调试训练"""
    
    setup_debug_environment()
    
    print("🚀 调试训练启动器")
    print("=" * 50)
    
    # 显示GPU信息
    gpu_count = torch.cuda.device_count()
    print(f"🎮 检测到 {gpu_count} 个GPU")
    
    if torch.cuda.is_available():
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            print(f"   GPU {i}: {props.name} - {props.total_memory / 1024**3:.1f} GB")
    
    # 逐步测试
    print("\n" + "="*50)
    print("开始逐步调试...")
    
    # 1. 测试数据加载
    if not test_data_loading():
        print("❌ 数据加载测试失败")
        return False
    
    # 2. 测试模型
    if not test_model_forward():
        print("❌ 模型测试失败")
        return False
    
    # 3. 测试Accelerate
    if not test_accelerate():
        print("❌ Accelerate测试失败")
        return False
    
    print("\n" + "="*50)
    print("✅ 所有组件测试通过!")
    print("💡 问题可能在训练循环的具体实现中")
    
    # 启动简化训练测试
    print("\n🔄 启动简化训练测试...")
    
    cmd = [
        "python", "-u",
        "/kaggle/working/VAE/simple_train.py"
    ]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # 实时输出
        for line in process.stdout:
            print(line.rstrip())
        
        return_code = process.wait()
        
        if return_code == 0:
            print("\n✅ 简化训练测试成功!")
            return True
        else:
            print(f"\n❌ 简化训练测试失败 (退出码: {return_code})")
            return False
            
    except Exception as e:
        print(f"\n❌ 简化训练启动失败: {e}")
        return False

def main():
    """主函数"""
    success = launch_debug_training()
    
    if success:
        print("\n🎉 调试完成!")
        print("💡 可以尝试运行完整训练")
    else:
        print("\n❌ 调试发现问题!")
        print("💡 请查看上面的错误信息")

if __name__ == "__main__":
    main()
