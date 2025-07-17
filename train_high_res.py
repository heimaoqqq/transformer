#!/usr/bin/env python3
"""
高分辨率训练启动器 - 256×256分辨率
充分利用双GPU T4的显存资源
"""

import os
import sys
import subprocess
import torch
from pathlib import Path

def setup_high_res_environment():
    """设置高分辨率训练环境"""
    # 优化内存分配
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
    
    # 设置无缓冲输出
    os.environ['PYTHONUNBUFFERED'] = '1'
    
    # 启用优化
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

def launch_high_res_training():
    """启动高分辨率训练"""
    
    setup_high_res_environment()
    
    gpu_count = torch.cuda.device_count()
    print(f"🎮 检测到 {gpu_count} 个GPU")
    
    # 显示GPU内存信息
    if torch.cuda.is_available():
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            print(f"   GPU {i}: {props.name} - {props.total_memory / 1024**3:.1f} GB")
            
            # 清理GPU缓存
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()
    
    if gpu_count > 1:
        print("🚀 启动双GPU高分辨率训练 (256×256)...")
        
        # 设置多GPU环境变量
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
            "--batch_size", "4",        # 双GPU总共4个样本 (保守)
            "--num_epochs", "40",       # 充分训练
            "--learning_rate", "0.0001",
            "--mixed_precision", "fp16",
            "--gradient_accumulation_steps", "4",
            "--kl_weight", "1e-6",
            "--perceptual_weight", "0.0",  # 暂时禁用感知损失
            "--freq_weight", "0.05",
            "--resolution", "256",      # 高分辨率
            "--num_workers", "1",       # 减少线程数节省内存
            "--save_interval", "10",
            "--log_interval", "5",
            "--sample_interval", "100", # 正常采样频率
            "--experiment_name", "kaggle_vae_256"
        ]
        
        print(f"📊 配置: batch_size=4 (每GPU 2个), resolution=256×256")
        
    else:
        print("🚀 启动单GPU高分辨率训练 (256×256)...")
        
        cmd = [
            "python", "-u",
            "training/train_vae.py",
            "--data_dir", "/kaggle/input/dataset",
            "--output_dir", "/kaggle/working/outputs/vae",
            "--batch_size", "2",        # 单GPU 2个样本 (保守)
            "--num_epochs", "40",
            "--learning_rate", "0.0001",
            "--mixed_precision", "fp16",
            "--gradient_accumulation_steps", "8",
            "--kl_weight", "1e-6",
            "--perceptual_weight", "0.0",
            "--freq_weight", "0.05",
            "--resolution", "256",
            "--num_workers", "1",
            "--save_interval", "10",
            "--log_interval", "5",
            "--sample_interval", "100",
            "--experiment_name", "kaggle_vae_256"
        ]
        
        print(f"📊 配置: batch_size=2, resolution=256×256")
    
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
            print("\n✅ 高分辨率训练完成!")
            return True
        else:
            print(f"\n❌ 训练失败 (退出码: {return_code})")
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
    print("🚀 Kaggle高分辨率VAE训练 (256×256)")
    print("=" * 50)
    
    success = launch_high_res_training()
    
    if success:
        print("\n🎉 高分辨率训练完成!")
        print("📁 模型保存在: /kaggle/working/outputs/vae/final_model")
    else:
        print("\n❌ 训练失败!")
        print("💡 如果内存不足，可以:")
        print("   - 减少batch_size")
        print("   - 降低分辨率到192或128")
        print("   - 增加gradient_accumulation_steps")
        sys.exit(1)

if __name__ == "__main__":
    main()
