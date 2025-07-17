#!/usr/bin/env python3
"""
带实时进度显示的训练启动器
确保在Kaggle环境中正确显示训练进度
"""

import os
import sys
import subprocess
import torch
from pathlib import Path

def setup_progress_environment():
    """设置进度显示环境"""
    # 确保输出不被缓冲
    os.environ['PYTHONUNBUFFERED'] = '1'
    
    # 设置终端环境
    os.environ['TERM'] = 'xterm-256color'
    
    # 强制启用颜色输出
    os.environ['FORCE_COLOR'] = '1'
    os.environ['CLICOLOR_FORCE'] = '1'

def launch_vae_training():
    """启动VAE训练"""
    
    setup_progress_environment()
    
    gpu_count = torch.cuda.device_count()
    print(f"🎮 检测到 {gpu_count} 个GPU")
    
    if gpu_count > 1:
        print("🚀 启动多GPU训练...")
        
        # 设置多GPU环境变量
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(gpu_count))
        os.environ['WORLD_SIZE'] = str(gpu_count)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        # 使用accelerate launch
        cmd = [
            "accelerate", "launch",
            "--config_file", "accelerate_config.yaml",
            "--num_processes", str(gpu_count),
            "training/train_vae.py",
            "--data_dir", "/kaggle/input/dataset",
            "--output_dir", "/kaggle/working/outputs/vae",
            "--batch_size", "6",
            "--num_epochs", "40",
            "--learning_rate", "0.0001",
            "--mixed_precision", "fp16",
            "--gradient_accumulation_steps", "2",
            "--kl_weight", "1e-6",
            "--perceptual_weight", "0.0",
            "--freq_weight", "0.05",
            "--resolution", "256",
            "--num_workers", "2",
            "--save_interval", "10",
            "--log_interval", "5",
            "--sample_interval", "100",
            "--experiment_name", "kaggle_vae"
        ]
    else:
        print("🚀 启动单GPU训练...")
        
        cmd = [
            "python", "-u",  # -u 确保无缓冲输出
            "training/train_vae.py",
            "--data_dir", "/kaggle/input/dataset",
            "--output_dir", "/kaggle/working/outputs/vae",
            "--batch_size", "4",
            "--num_epochs", "40",
            "--learning_rate", "0.0001",
            "--mixed_precision", "fp16",
            "--gradient_accumulation_steps", "4",
            "--kl_weight", "1e-6",
            "--perceptual_weight", "0.0",
            "--freq_weight", "0.05",
            "--resolution", "128",
            "--num_workers", "1",
            "--save_interval", "10",
            "--log_interval", "5",
            "--sample_interval", "200",
            "--experiment_name", "kaggle_vae"
        ]
    
    print(f"Command: {' '.join(cmd)}")
    print("=" * 80)
    
    try:
        # 使用实时输出，确保进度条正常显示
        process = subprocess.Popen(
            cmd,
            stdout=None,  # 直接输出到终端
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=0  # 无缓冲
        )
        
        # 等待完成
        return_code = process.wait()
        
        if return_code == 0:
            print("\n✅ VAE训练完成!")
            return True
        else:
            print(f"\n❌ VAE训练失败 (退出码: {return_code})")
            return False
            
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
        process.terminate()
        return False
    except Exception as e:
        print(f"\n❌ 训练启动失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 Kaggle VAE训练 (带进度显示)")
    print("=" * 50)
    
    success = launch_vae_training()
    
    if success:
        print("\n🎉 训练完成!")
    else:
        print("\n❌ 训练失败!")
        sys.exit(1)

if __name__ == "__main__":
    main()
