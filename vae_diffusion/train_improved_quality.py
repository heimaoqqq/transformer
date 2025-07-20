#!/usr/bin/env python3
"""
改进的VAE训练器 - 降低压缩比，提升重建质量
64×64 → 16×16 (压缩比12:1)
"""

import os
import sys
import subprocess
import torch
from pathlib import Path

def setup_environment():
    """设置环境"""
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
    os.environ['PYTHONUNBUFFERED'] = '1'
    torch.backends.cudnn.benchmark = True

def get_improved_config():
    """获取改进配置 - 降低压缩比"""
    
    if not torch.cuda.is_available():
        print("❌ 需要CUDA支持")
        return None

    gpu_name = torch.cuda.get_device_properties(0).name
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f}GB)")

    # 128×128高分辨率配置 (需要更多显存)
    if "P100" in gpu_name or gpu_memory > 14:
        return {
            "batch_size": 8,            # 128×128需要降低批次
            "mixed_precision": "no",
            "learning_rate": "0.0001",
            "gradient_accumulation": 2,  # 增加累积保持有效批次
            "num_workers": 2,
        }
    elif "T4" in gpu_name or gpu_memory > 10:
        return {
            "batch_size": 6,            # T4进一步降低
            "mixed_precision": "fp16",
            "learning_rate": "0.0001",
            "gradient_accumulation": 3,  # 更多累积
            "num_workers": 2,
        }
    else:
        return {
            "batch_size": 4,            # 低端GPU最小批次
            "mixed_precision": "fp16",
            "learning_rate": "0.0001",
            "gradient_accumulation": 4,  # 最大累积
            "num_workers": 1,
        }

def launch_improved_training():
    """启动改进的训练"""
    setup_environment()
    
    config = get_improved_config()
    if not config:
        return False
    
    print(f"🚀 启动现代化VAE训练 (128×128 → 32×32)")

    # 有效批次大小
    effective_batch = config["batch_size"] * config["gradient_accumulation"]
    print(f"📊 配置: 批次{config['batch_size']} × 累积{config['gradient_accumulation']} = 有效批次{effective_batch}")
    print(f"⚙️  参数: 学习率{config['learning_rate']}, 精度{config['mixed_precision']}")
    print(f"🎯 目标: 128×128→32×32×4, 压缩比12:1, PSNR>28dB")
    print(f"🖼️  模式: 现代化高质量训练 (Lanczos缩放)")
    
    # 构建命令
    cmd = [
        "python", 
        "training/train_vae.py",
        "--data_dir", "/kaggle/input/dataset",
        "--output_dir", "/kaggle/working/outputs/vae_improved_quality",
        "--batch_size", str(config["batch_size"]),
        "--num_epochs", "80",  # 保持80轮
        "--learning_rate", config["learning_rate"],
        "--mixed_precision", config["mixed_precision"],
        "--gradient_accumulation_steps", str(config["gradient_accumulation"]),
        "--kl_weight", "1e-6",  # 保持低KL权重
        "--perceptual_weight", "0.5",  # 增加感知损失权重
        "--freq_weight", "0.0",  # 禁用频域损失
        "--resolution", "128",  # 升级到128×128输入分辨率
        "--num_workers", str(config["num_workers"]),
        "--save_interval", "5",
        "--log_interval", "2",
        "--sample_interval", "50",

        # 关键: 现代化架构参数 (128×128 → 32×32)
        "--down_block_types", "DownEncoderBlock2D,DownEncoderBlock2D,DownEncoderBlock2D",  # 3层下采样: 128→64→32
        "--up_block_types", "UpDecoderBlock2D,UpDecoderBlock2D,UpDecoderBlock2D",        # 3层上采样: 32→64→128
        "--block_out_channels", "128,256,512",                               # 3层通道配置
        "--layers_per_block", "1",                                       # 每层1个ResNet块 (标准配置)
        "--latent_channels", "4",                                        # 保持4通道
        "--sample_size", "128",                                          # 修复: 设置sample_size为128匹配输入尺寸
    ]
    
    print(f"\n🏗️  现代化架构 (128×128 → 32×32):")
    print(f"   📐 输入: 128×128×3 = 49,152 像素")
    print(f"   🔽 下采样: 128→64→32 (3层)")
    print(f"   � 通道数: [128, 256, 512] (现代标准)")
    print(f"   🧱 每层块数: 1 (标准配置)")
    print(f"   🎯 潜在空间: 32×32×4 = 4,096 维度")
    print(f"   📊 压缩比: 12:1 (vs 之前48:1)")
    print(f"   🖼️  缩放方法: Lanczos (现代高质量)")

    print(f"\n📈 预期显著改进:")
    print(f"   💾 显存使用: ~6-8GB (vs 3GB)")
    print(f"   ⏱️  训练时间: ~15-25分钟/轮 (vs 10分钟)")
    print(f"   📦 批次大小: {config['batch_size']} (适应更大输入)")
    print(f"   🎯 PSNR目标: 28+ dB (vs 21.78 dB)")
    print(f"   ✨ 信息容量: 4倍提升 (4K vs 1K)")
    print(f"   🔍 细节保留: 显著提升")
    
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
            print(f"\n✅ 改进VAE训练完成!")
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
    print("🎨 现代化VAE训练 (128×128 → 32×32)")

    print("📊 数据集: ~5000张图像, 31用户")
    print("🎯 目标: 显著提升重建质量和细节保留")
    print("🔧 策略: 现代化架构 + Lanczos缩放 + 4倍信息容量")
    
    success = launch_improved_training()
    
    if success:
        print("\n🎉 现代化VAE训练完成!")
        print("📁 模型: /kaggle/working/outputs/vae_improved_quality/final_model")
        print("🔍 质量检查: python check_vae.py --model_path /kaggle/working/outputs/vae_improved_quality/final_model")
        print("📊 配置测试: python test_128x128_config.py")
        print("🎯 预期PSNR: 28+ dB (vs 之前21.78 dB)")
    else:
        print("\n❌ 训练失败! 请检查GPU内存和数据路径")
        print("💡 建议: 先运行 python test_128x128_config.py 测试配置")
        sys.exit(1)

if __name__ == "__main__":
    main()
