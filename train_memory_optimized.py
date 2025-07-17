#!/usr/bin/env python3
"""
内存优化的训练启动器
专门针对Kaggle双GPU T4环境的内存限制
"""

import os
import sys
import subprocess
import torch
from pathlib import Path

def setup_memory_environment():
    """设置内存优化环境"""
    # 设置PyTorch内存管理
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    # 设置无缓冲输出
    os.environ['PYTHONUNBUFFERED'] = '1'
    
    # 启用内存映射
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

def get_optimal_config():
    """根据GPU内存获取最优配置"""

    gpu_count = torch.cuda.device_count()

    if gpu_count >= 2:
        # 双GPU配置 - 256分辨率优化
        return {
            "batch_size": 6,      # 每GPU 3个样本，总共6个
            "resolution": 256,    # 恢复256分辨率
            "num_workers": 2,     # 多线程
            "gradient_accumulation_steps": 2,  # 正常梯度累积
        }
    else:
        # 单GPU配置
        return {
            "batch_size": 4,      # 单GPU更大批次
            "resolution": 256,    # 256分辨率
            "num_workers": 2,     # 多线程
            "gradient_accumulation_steps": 4, # 适中梯度累积
        }

def launch_optimized_training():
    """启动内存优化训练"""
    
    setup_memory_environment()
    
    gpu_count = torch.cuda.device_count()
    print(f"🎮 检测到 {gpu_count} 个GPU")
    
    # 清理GPU缓存
    if torch.cuda.is_available():
        for i in range(gpu_count):
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()
    
    config = get_optimal_config()
    print(f"📊 使用配置: {config}")
    
    if gpu_count > 1:
        print("🚀 启动双GPU内存优化训练...")
        
        # 设置环境变量
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
        os.environ['WORLD_SIZE'] = str(gpu_count)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        cmd = [
            "accelerate", "launch",
            "--config_file", "accelerate_config.yaml",
            "--num_processes", str(gpu_count),
            "training/train_vae.py",
            "--data_dir", "/kaggle/input/dataset",
            "--output_dir", "/kaggle/working/outputs/vae",
            "--batch_size", str(config["batch_size"]),
            "--num_epochs", "40",  # 恢复正常epoch数
            "--learning_rate", "0.0001",
            "--mixed_precision", "fp16",
            "--gradient_accumulation_steps", str(config["gradient_accumulation_steps"]),
            "--kl_weight", "1e-6",
            "--perceptual_weight", "0.0",  # 禁用感知损失
            "--freq_weight", "0.05",
            "--resolution", str(config["resolution"]),
            "--num_workers", str(config["num_workers"]),
            "--save_interval", "10",
            "--log_interval", "5",
            "--sample_interval", "100",  # 恢复正常采样
            "--experiment_name", "kaggle_vae_optimized"
        ]
    else:
        print("🚀 启动单GPU内存优化训练...")
        
        cmd = [
            "python", "-u",
            "training/train_vae.py",
            "--data_dir", "/kaggle/input/dataset",
            "--output_dir", "/kaggle/working/outputs/vae",
            "--batch_size", str(config["batch_size"]),
            "--num_epochs", "40",
            "--learning_rate", "0.0001",
            "--mixed_precision", "fp16",
            "--gradient_accumulation_steps", str(config["gradient_accumulation_steps"]),
            "--kl_weight", "1e-6",
            "--perceptual_weight", "0.0",
            "--freq_weight", "0.05",
            "--resolution", str(config["resolution"]),
            "--num_workers", str(config["num_workers"]),
            "--save_interval", "10",
            "--log_interval", "5",
            "--sample_interval", "100",
            "--experiment_name", "kaggle_vae_optimized"
        ]
    
    print(f"Command: {' '.join(cmd)}")
    print("=" * 80)
    
    try:
        # 启动训练
        process = subprocess.Popen(
            cmd,
            stdout=None,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=0
        )
        
        return_code = process.wait()
        
        if return_code == 0:
            print("\n✅ 内存优化训练完成!")
            return True
        else:
            print(f"\n❌ 训练失败 (退出码: {return_code})")
            return False
            
    except Exception as e:
        print(f"\n❌ 训练启动失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 Kaggle内存优化VAE训练")
    print("=" * 50)
    
    # 显示内存信息
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name} - {props.total_memory / 1024**3:.1f} GB")
    
    success = launch_optimized_training()
    
    if success:
        print("\n🎉 训练完成!")
    else:
        print("\n❌ 训练失败!")
        print("💡 建议: 尝试更小的批次大小或分辨率")
        sys.exit(1)

if __name__ == "__main__":
    main()
