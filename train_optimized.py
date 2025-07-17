#!/usr/bin/env python3
"""
优化版VAE训练启动器
针对速度和效率优化
"""

import os
import sys
import subprocess
import torch
from pathlib import Path

def setup_optimized_environment():
    """设置优化环境"""
    # 优化内存分配
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    # 启用优化
    os.environ['PYTHONUNBUFFERED'] = '1'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    
    # 启用cuDNN优化
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

def launch_optimized_training():
    """启动优化训练"""
    
    setup_optimized_environment()
    
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
        print("🚀 启动优化双GPU训练...")
        
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
            "--batch_size", "8",        # 增加批次大小
            "--num_epochs", "30",       # 减少epoch数
            "--learning_rate", "0.0002", # 稍微提高学习率
            "--mixed_precision", "fp16",
            "--gradient_accumulation_steps", "2", # 减少梯度累积
            "--kl_weight", "1e-6",
            "--perceptual_weight", "0.0",
            "--freq_weight", "0.05",
            "--resolution", "256",
            "--num_workers", "2",       # 增加数据加载线程
            "--save_interval", "5",     # 更频繁保存
            "--log_interval", "2",      # 更频繁日志
            "--sample_interval", "50",  # 更频繁采样
            "--experiment_name", "kaggle_vae_optimized"
        ]
        
        print(f"📊 优化配置:")
        print(f"   批次大小: 8 (每GPU 4个)")
        print(f"   分辨率: 256×256")
        print(f"   压缩: 256→32 (8倍下采样)")
        print(f"   数据线程: 2")
        print(f"   混合精度: FP16")
        
    else:
        print("🚀 启动优化单GPU训练...")
        
        cmd = [
            "python", "-u",
            "training/train_vae.py",
            "--data_dir", "/kaggle/input/dataset",
            "--output_dir", "/kaggle/working/outputs/vae",
            "--batch_size", "6",        # 单GPU更大批次
            "--num_epochs", "30",
            "--learning_rate", "0.0002",
            "--mixed_precision", "fp16",
            "--gradient_accumulation_steps", "3",
            "--kl_weight", "1e-6",
            "--perceptual_weight", "0.0",
            "--freq_weight", "0.05",
            "--resolution", "256",
            "--num_workers", "2",
            "--save_interval", "5",
            "--log_interval", "2",
            "--sample_interval", "50",
            "--experiment_name", "kaggle_vae_optimized"
        ]
        
        print(f"📊 优化配置:")
        print(f"   批次大小: 6")
        print(f"   分辨率: 256×256")
        print(f"   数据线程: 2")
    
    print(f"\nCommand: {' '.join(cmd)}")
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
            print("\n✅ 优化训练完成!")
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
    print("🚀 优化版VAE训练")
    print("=" * 50)
    
    print("🎯 优化重点:")
    print("   ✅ 增加批次大小 (4→8)")
    print("   ✅ 增加数据加载线程 (1→2)")
    print("   ✅ 启用cuDNN优化")
    print("   ✅ 优化内存分配")
    print("   ✅ 减少epoch数 (40→30)")
    print("   ✅ 提高学习率 (1e-4→2e-4)")
    
    print("\n📊 预期改进:")
    print("   🚀 训练速度: +50-80%")
    print("   ⏱️  每轮时间: 30分钟→15-20分钟")
    print("   💾 内存使用: ~12-14GB")
    
    success = launch_optimized_training()
    
    if success:
        print("\n🎉 优化训练完成!")
        print("📁 模型保存在: /kaggle/working/outputs/vae/final_model")
    else:
        print("\n❌ 训练失败!")
        print("💡 如果内存不足，可以:")
        print("   - 减少batch_size到6")
        print("   - 减少num_workers到1")
        print("   - 使用train_high_res.py")
        sys.exit(1)

if __name__ == "__main__":
    main()
