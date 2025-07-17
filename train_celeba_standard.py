#!/usr/bin/env python3
"""
CelebA标准微多普勒VAE训练器
采用64×64分辨率，遵循成熟项目的标准做法
"""

import os
import sys
import subprocess
import torch
from pathlib import Path

def setup_celeba_environment():
    """设置CelebA标准环境"""
    # 强制单GPU (CelebA标准通常单GPU训练)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # 优化内存分配
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
    os.environ['PYTHONUNBUFFERED'] = '1'
    
    # 基础优化
    torch.backends.cudnn.benchmark = True

def get_celeba_standard_config():
    """获取配置 - 针对小数据集优化"""

    if not torch.cuda.is_available():
        print("❌ 需要CUDA支持")
        return None

    gpu_name = torch.cuda.get_device_properties(0).name
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f}GB)")

    # 小数据集优化配置 (5000张图像)
    if "P100" in gpu_name or gpu_memory > 14:
        return {
            "batch_size": 16,           # 适合小数据集
            "mixed_precision": "no",
            "learning_rate": "0.0001",
            "gradient_accumulation": 1,
            "num_workers": 2,
        }
    elif "T4" in gpu_name or gpu_memory > 10:
        return {
            "batch_size": 12,           # T4适中批次
            "mixed_precision": "fp16",
            "learning_rate": "0.0001",
            "gradient_accumulation": 2,
            "num_workers": 2,
        }
    else:
        return {
            "batch_size": 8,            # 保守批次
            "mixed_precision": "fp16",
            "learning_rate": "0.0001",
            "gradient_accumulation": 2,
            "num_workers": 1,
        }

def launch_celeba_training():
    """启动CelebA标准训练"""
    
    setup_celeba_environment()
    
    # 获取配置
    config = get_celeba_standard_config()
    if config is None:
        return False
    
    # 清理GPU缓存
    torch.cuda.empty_cache()

    print(f"\n🚀 启动VAE训练 (批次:{config['batch_size']}, 精度:{config['mixed_precision']})")
    
    cmd = [
        "python", "-u",
        "training/train_vae.py",
        "--data_dir", "/kaggle/input/dataset",
        "--output_dir", "/kaggle/working/outputs/vae_celeba_standard",
        "--batch_size", str(config["batch_size"]),
        "--num_epochs", "80",  # 增加训练轮数，配合低学习率
        "--learning_rate", config["learning_rate"],
        "--mixed_precision", config["mixed_precision"],
        "--gradient_accumulation_steps", str(config["gradient_accumulation"]),
        "--kl_weight", "1e-6",  # 降低KL权重，避免过度正则化
        "--perceptual_weight", "0.1",  # 降低感知损失权重，避免设备问题
        "--freq_weight", "0.05",  # 微多普勒特有
        "--resolution", "64",  # CelebA标准分辨率
        "--num_workers", str(config["num_workers"]),
        "--save_interval", "5",
        "--log_interval", "2",
        "--sample_interval", "50",
        "--experiment_name", "micro_doppler_celeba_standard"
    ]
    
    print(f"📊 CelebA标准配置:")
    # 有效批次大小
    effective_batch = config["batch_size"] * config["gradient_accumulation"]
    print(f"📊 配置: 批次{config['batch_size']} × 累积{config['gradient_accumulation']} = 有效批次{effective_batch}")
    print(f"⚙️  参数: 学习率{config['learning_rate']}, 精度{config['mixed_precision']}, 线程{config['num_workers']}")
    print(f"🎯 目标: 64×64→8×8×4, 压缩比48:1, PSNR>25dB")
    
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
            print(f"\n✅ VAE训练完成!")
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
    print("🎨 微多普勒VAE训练 (小数据集优化)")

    print("📊 数据集: ~5000张图像, 31用户")
    print("🎯 目标: 64×64→8×8×4, PSNR>25dB")

    success = launch_celeba_training()

    if success:
        print("\n🎉 训练完成!")
        print("📁 模型: /kaggle/working/outputs/vae_celeba_standard/final_model")
        print("� 检查质量: python check_vae.py")
    else:
        print("\n❌ 训练失败! 请检查GPU内存和数据路径")
        sys.exit(1)

if __name__ == "__main__":
    main()
