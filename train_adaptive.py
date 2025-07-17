#!/usr/bin/env python3
"""
GPU自适应训练器
- P100: FP32 + 大批次
- T4: FP16 + 适中批次
"""

import os
import sys
import subprocess
import torch
from pathlib import Path

def detect_gpu_type():
    """检测GPU类型"""
    if not torch.cuda.is_available():
        return None
    
    gpu_name = torch.cuda.get_device_properties(0).name
    
    if "P100" in gpu_name:
        return "P100"
    elif "T4" in gpu_name:
        return "T4"
    else:
        return "Unknown"

def get_gpu_optimized_config(gpu_type):
    """根据GPU类型获取优化配置"""
    
    if gpu_type == "P100":
        return {
            "batch_size": 8,            # P100保守批次
            "mixed_precision": "no",    # P100没有Tensor Core
            "learning_rate": "0.0002",  # 适中学习率
            "gradient_accumulation": 2, # 适中累积
            "num_workers": 2,           # 适中线程
            "memory_efficient": True,   # 保守内存管理
        }
    elif gpu_type == "T4":
        return {
            "batch_size": 4,            # T4保守批次
            "mixed_precision": "fp16",  # T4有Tensor Core
            "learning_rate": "0.00015", # 降低学习率
            "gradient_accumulation": 4, # 增加累积保持有效批次
            "num_workers": 1,           # 单线程避免竞争
            "memory_efficient": True,   # T4需要节省内存
        }
    else:
        # 保守配置
        return {
            "batch_size": 4,
            "mixed_precision": "no",
            "learning_rate": "0.0001",
            "gradient_accumulation": 4,
            "num_workers": 1,
            "memory_efficient": True,
        }

def setup_gpu_environment(gpu_type, config):
    """设置GPU环境"""
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 单GPU避免通信开销
    os.environ['PYTHONUNBUFFERED'] = '1'
    
    if config["memory_efficient"]:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    else:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    # GPU特定优化
    if gpu_type == "P100":
        # P100优化：FP32性能强
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    elif gpu_type == "T4":
        # T4优化：Tensor Core
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True

def launch_adaptive_training():
    """启动自适应训练"""
    
    # 检测GPU类型
    gpu_type = detect_gpu_type()
    
    if gpu_type is None:
        print("❌ 没有检测到CUDA GPU")
        return False
    
    print(f"🎮 检测到GPU: {gpu_type}")
    
    # 获取优化配置
    config = get_gpu_optimized_config(gpu_type)
    
    # 设置环境
    setup_gpu_environment(gpu_type, config)
    
    # 显示GPU信息
    props = torch.cuda.get_device_properties(0)
    print(f"   名称: {props.name}")
    print(f"   内存: {props.total_memory / 1024**3:.1f} GB")
    print(f"   计算能力: {props.major}.{props.minor}")
    
    # 清理GPU缓存
    torch.cuda.empty_cache()
    
    print(f"\n🚀 启动{gpu_type}优化训练...")
    
    cmd = [
        "python", "-u",
        "training/train_vae.py",
        "--data_dir", "/kaggle/input/dataset",
        "--output_dir", "/kaggle/working/outputs/vae",
        "--batch_size", str(config["batch_size"]),
        "--num_epochs", "25",  # 适中epoch数
        "--learning_rate", config["learning_rate"],
        "--mixed_precision", config["mixed_precision"],
        "--gradient_accumulation_steps", str(config["gradient_accumulation"]),
        "--kl_weight", "1e-6",
        "--perceptual_weight", "0.0",
        "--freq_weight", "0.05",
        "--resolution", "256",
        "--num_workers", str(config["num_workers"]),
        "--save_interval", "5",
        "--log_interval", "2",
        "--sample_interval", "50",
        "--experiment_name", f"kaggle_vae_{gpu_type.lower()}"
    ]
    
    print(f"📊 {gpu_type}优化配置:")
    print(f"   📦 批次大小: {config['batch_size']}")
    print(f"   🔢 混合精度: {config['mixed_precision']}")
    print(f"   📈 学习率: {config['learning_rate']}")
    print(f"   ⚡ 梯度累积: {config['gradient_accumulation']}")
    print(f"   🧵 数据线程: {config['num_workers']}")
    print(f"   💾 内存优化: {config['memory_efficient']}")
    
    # 预期性能
    if gpu_type == "P100":
        print(f"\n📊 P100预期性能:")
        print(f"   🚀 FP32高性能计算")
        print(f"   💾 大内存支持大批次")
        print(f"   ⏱️  预期每轮: 8-12分钟")
        print(f"   🎯 总训练时间: 3-5小时")
    elif gpu_type == "T4":
        print(f"\n📊 T4预期性能:")
        print(f"   🚀 FP16 Tensor Core加速")
        print(f"   💾 内存节省50%")
        print(f"   ⏱️  预期每轮: 12-18分钟")
        print(f"   🎯 总训练时间: 5-7小时")
    
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
            print(f"\n✅ {gpu_type}优化训练完成!")
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
    print("🎯 GPU自适应VAE训练")
    print("=" * 50)
    
    gpu_type = detect_gpu_type()
    
    if gpu_type == "P100":
        print("🎯 P100优化策略:")
        print("   ✅ FP32精度 (P100没有Tensor Core)")
        print("   ✅ 大批次训练 (16GB内存)")
        print("   ✅ 高学习率 (配合大批次)")
        print("   ✅ 无梯度累积 (性能强)")
        print("   ✅ 多线程数据加载")
        
    elif gpu_type == "T4":
        print("🎯 T4优化策略:")
        print("   ✅ FP16混合精度 (Tensor Core)")
        print("   ✅ 适中批次 (15GB内存)")
        print("   ✅ 内存优化")
        print("   ✅ 适中梯度累积")
        
    print("\n🏗️  通用优化:")
    print("   ✅ 3层下采样 (55M参数)")
    print("   ✅ 单GPU避免通信开销")
    print("   ✅ 256×256高分辨率")
    print("   ✅ 32×32×4潜在空间")
    
    success = launch_adaptive_training()
    
    if success:
        print(f"\n🎉 {gpu_type}优化训练完成!")
        print("📁 模型保存在: /kaggle/working/outputs/vae/final_model")
    else:
        print("\n❌ 训练失败!")
        sys.exit(1)

if __name__ == "__main__":
    main()
