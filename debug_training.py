#!/usr/bin/env python3
"""
调试训练脚本 - 找出训练卡住的原因
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import time

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir
sys.path.insert(0, str(project_root))

def test_gpu_setup():
    """测试GPU设置"""
    print("🔍 测试GPU设置...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"✅ 检测到 {gpu_count} 个GPU")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        print(f"   GPU {i}: {props.name} - {props.total_memory / 1024**3:.1f} GB")
    
    return True

def test_data_loading():
    """测试数据加载"""
    print("\n🔍 测试数据加载...")
    
    try:
        from utils.data_loader import MicroDopplerDataset
        from torch.utils.data import DataLoader
        
        # 创建数据集
        dataset = MicroDopplerDataset(
            data_dir="/kaggle/input/dataset",
            resolution=256,
            split="train"
        )
        
        print(f"✅ 数据集大小: {len(dataset)}")
        
        # 创建数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=2,  # 小批次测试
            shuffle=False,
            num_workers=0,  # 单线程测试
            pin_memory=True
        )
        
        print("✅ 数据加载器创建成功")
        
        # 测试加载一个批次
        print("🔄 测试加载第一个批次...")
        start_time = time.time()
        
        for i, batch in enumerate(dataloader):
            load_time = time.time() - start_time
            print(f"✅ 批次 {i} 加载成功 ({load_time:.2f}s)")
            print(f"   图像形状: {batch['image'].shape}")
            print(f"   用户索引: {batch['user_idx'].shape}")
            
            if i >= 2:  # 只测试前3个批次
                break
            
            start_time = time.time()
        
        return True
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_creation():
    """测试模型创建"""
    print("\n🔍 测试模型创建...")
    
    try:
        from diffusers import AutoencoderKL
        
        print("🔄 创建VAE模型...")
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
        
        print("✅ VAE模型创建成功")
        
        # 移动到GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vae = vae.to(device)
        print(f"✅ 模型移动到 {device}")
        
        return vae, device
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_forward_pass(vae, device):
    """测试前向传播"""
    print("\n🔍 测试前向传播...")
    
    try:
        # 创建测试数据
        batch_size = 2
        test_images = torch.randn(batch_size, 3, 256, 256).to(device)
        print(f"✅ 测试数据创建成功: {test_images.shape}")
        
        # 测试编码
        print("🔄 测试VAE编码...")
        start_time = time.time()
        
        with torch.no_grad():
            posterior = vae.encode(test_images).latent_dist
            latents = posterior.sample()
        
        encode_time = time.time() - start_time
        print(f"✅ 编码成功 ({encode_time:.2f}s): {latents.shape}")
        
        # 测试解码
        print("🔄 测试VAE解码...")
        start_time = time.time()
        
        with torch.no_grad():
            reconstruction = vae.decode(latents).sample
        
        decode_time = time.time() - start_time
        print(f"✅ 解码成功 ({decode_time:.2f}s): {reconstruction.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_accelerate_setup():
    """测试Accelerate设置"""
    print("\n🔍 测试Accelerate设置...")
    
    try:
        from accelerate import Accelerator
        
        accelerator = Accelerator(
            gradient_accumulation_steps=2,
            mixed_precision="fp16"
        )
        
        print(f"✅ Accelerator创建成功")
        print(f"   设备: {accelerator.device}")
        print(f"   进程数: {accelerator.num_processes}")
        print(f"   分布式类型: {accelerator.distributed_type}")
        print(f"   是否主进程: {accelerator.is_main_process}")
        
        return accelerator
        
    except Exception as e:
        print(f"❌ Accelerate设置失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_training_step():
    """测试完整的训练步骤"""
    print("\n🔍 测试完整训练步骤...")
    
    # 测试所有组件
    if not test_gpu_setup():
        return False
    
    if not test_data_loading():
        return False
    
    vae, device = test_model_creation()
    if vae is None:
        return False
    
    if not test_forward_pass(vae, device):
        return False
    
    accelerator = test_accelerate_setup()
    if accelerator is None:
        return False
    
    print("\n🎉 所有测试通过！")
    return True

def main():
    """主函数"""
    print("🚀 训练调试工具")
    print("=" * 50)
    
    success = test_training_step()
    
    if success:
        print("\n✅ 调试完成 - 没有发现问题")
        print("💡 建议: 尝试减少批次大小或检查数据集")
    else:
        print("\n❌ 发现问题 - 请查看上面的错误信息")

if __name__ == "__main__":
    main()
