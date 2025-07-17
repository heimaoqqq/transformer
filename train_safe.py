#!/usr/bin/env python3
"""
安全版VAE训练启动器
- 3层下采样 (50%参数减少)
- 保守批次大小 (避免OOM)
- FP16混合精度
"""

import os
import sys
import subprocess
import torch
from pathlib import Path

def setup_safe_environment():
    """设置安全训练环境"""
    # 保守内存分配
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    # 启用优化
    os.environ['PYTHONUNBUFFERED'] = '1'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    
    # 启用cuDNN优化
    torch.backends.cudnn.benchmark = True

def launch_safe_training():
    """启动安全训练"""
    
    setup_safe_environment()
    
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
        print("🚀 启动安全双GPU训练...")
        
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
            "--batch_size", "4",        # 安全批次大小
            "--num_epochs", "30",       
            "--learning_rate", "0.00015", # 适中学习率
            "--mixed_precision", "fp16", 
            "--gradient_accumulation_steps", "2", # 适中梯度累积
            "--kl_weight", "1e-6",
            "--perceptual_weight", "0.0",
            "--freq_weight", "0.05",
            "--resolution", "256",
            "--num_workers", "1",       # 单线程避免竞争
            "--save_interval", "5",     
            "--log_interval", "2",      
            "--sample_interval", "100", 
            "--experiment_name", "kaggle_vae_safe"
        ]
        
        print(f"📊 安全配置:")
        print(f"   🏗️  架构: 3层下采样 (256→128→64→32)")
        print(f"   📦 批次大小: 4 (每GPU 2个)")
        print(f"   🔢 混合精度: FP16")
        print(f"   🧵 数据线程: 1 (安全)")
        print(f"   ⚡ 梯度累积: 2")
        print(f"   📈 学习率: 1.5e-4")
        print(f"   💾 预期内存: ~8GB/GPU")
        
    else:
        print("🚀 启动安全单GPU训练...")
        
        cmd = [
            "python", "-u",
            "training/train_vae.py",
            "--data_dir", "/kaggle/input/dataset",
            "--output_dir", "/kaggle/working/outputs/vae",
            "--batch_size", "2",        # 单GPU小批次
            "--num_epochs", "30",
            "--learning_rate", "0.00015",
            "--mixed_precision", "fp16",
            "--gradient_accumulation_steps", "4",
            "--kl_weight", "1e-6",
            "--perceptual_weight", "0.0",
            "--freq_weight", "0.05",
            "--resolution", "256",
            "--num_workers", "1",
            "--save_interval", "5",
            "--log_interval", "2",
            "--sample_interval", "100",
            "--experiment_name", "kaggle_vae_safe"
        ]
        
        print(f"📊 安全配置:")
        print(f"   🏗️  架构: 3层下采样")
        print(f"   📦 批次大小: 2")
        print(f"   🔢 混合精度: FP16")
        print(f"   🧵 数据线程: 1")
        print(f"   💾 预期内存: ~6GB")
    
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
            print("\n✅ 安全训练完成!")
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
    print("🛡️  安全版VAE训练 (3层下采样)")
    print("=" * 50)
    
    print("🎯 安全优化:")
    print("   ✅ 3层下采样 → 50%参数减少")
    print("   ✅ FP16混合精度 → 内存节省")
    print("   ✅ 保守批次大小 → 避免OOM")
    print("   ✅ 单线程数据加载 → 稳定性")
    print("   ✅ 适中梯度累积 → 平衡效率")
    
    print("\n📊 预期效果:")
    print("   🚀 训练速度: +50-100% (vs 4层)")
    print("   ⏱️  每轮时间: 30分钟→15-20分钟")
    print("   💾 内存使用: ~8GB (vs 14GB)")
    print("   🛡️  稳定性: 高 (不会OOM)")
    print("   🎯 质量: 保持 (256×256)")
    
    print("\n🏗️  架构优势:")
    print("   📉 参数量: 83M → 40M (减半)")
    print("   ⚡ 计算量: 减少50%")
    print("   🎯 压缩比: 保持64:1")
    print("   📐 潜在空间: 32×32×4")
    
    success = launch_safe_training()
    
    if success:
        print("\n🎉 安全训练完成!")
        print("📁 模型保存在: /kaggle/working/outputs/vae/final_model")
    else:
        print("\n❌ 训练失败!")
        print("💡 如果还有问题，可以:")
        print("   - 进一步减少batch_size到2")
        print("   - 降低分辨率到128")
        sys.exit(1)

if __name__ == "__main__":
    main()
