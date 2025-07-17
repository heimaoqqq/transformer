#!/usr/bin/env python3
"""
单GPU训练启动器 - 避免多GPU通信开销
"""

import os
import sys
import subprocess
import torch
from pathlib import Path

def setup_single_gpu_environment():
    """设置单GPU环境"""
    # 强制使用单GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # 优化设置
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    os.environ['PYTHONUNBUFFERED'] = '1'
    
    # 启用优化
    torch.backends.cudnn.benchmark = True

def launch_single_gpu_training():
    """启动单GPU训练"""
    
    setup_single_gpu_environment()
    
    print("🚀 启动单GPU高效训练...")
    print("💡 避免多GPU通信开销")
    
    # 显示GPU信息
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"   🎮 使用GPU: {props.name} - {props.total_memory / 1024**3:.1f} GB")
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
    
    cmd = [
        "python", "-u",
        "training/train_vae.py",
        "--data_dir", "/kaggle/input/dataset",
        "--output_dir", "/kaggle/working/outputs/vae",
        "--batch_size", "8",        # 单GPU可以用更大批次
        "--num_epochs", "30",
        "--learning_rate", "0.0002",
        "--mixed_precision", "fp16",
        "--gradient_accumulation_steps", "1", # 实时更新
        "--kl_weight", "1e-6",
        "--perceptual_weight", "0.0",
        "--freq_weight", "0.05",
        "--resolution", "256",
        "--num_workers", "2",       # 多线程数据加载
        "--save_interval", "5",
        "--log_interval", "2",
        "--sample_interval", "50",
        "--experiment_name", "kaggle_vae_single_gpu"
    ]
    
    print(f"📊 单GPU配置:")
    print(f"   🏗️  架构: 3层下采样 (55M参数)")
    print(f"   📦 批次大小: 8")
    print(f"   🔢 混合精度: FP16")
    print(f"   🧵 数据线程: 2")
    print(f"   ⚡ 梯度累积: 1 (实时)")
    print(f"   💾 预期内存: ~4-6GB")
    print(f"   🚀 预期速度: 2-3倍提升")
    
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
            print("\n✅ 单GPU训练完成!")
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
    print("🎯 单GPU高效训练")
    print("=" * 50)
    
    print("🎯 优化策略:")
    print("   ✅ 避免多GPU通信开销")
    print("   ✅ 3层下采样 (55M参数)")
    print("   ✅ 大批次训练 (batch_size=8)")
    print("   ✅ 实时梯度更新")
    print("   ✅ FP16混合精度")
    
    print("\n📊 预期改进:")
    print("   🚀 训练速度: +200-300%")
    print("   ⏱️  每轮时间: 53分钟→15-20分钟")
    print("   💾 内存使用: ~6GB")
    print("   🎯 质量: 保持 (256×256)")
    
    print("\n💡 为什么单GPU可能更快:")
    print("   - 避免GPU间通信延迟")
    print("   - 减少同步开销")
    print("   - 简化内存管理")
    print("   - 降低系统复杂度")
    
    success = launch_single_gpu_training()
    
    if success:
        print("\n🎉 单GPU训练完成!")
        print("📁 模型保存在: /kaggle/working/outputs/vae/final_model")
    else:
        print("\n❌ 训练失败!")
        sys.exit(1)

if __name__ == "__main__":
    main()
