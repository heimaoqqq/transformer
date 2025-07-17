#!/usr/bin/env python3
"""
显示当前VAE训练配置
普通图像训练模式
"""

import torch

def show_current_config():
    """显示当前训练配置"""
    print("🎨 VAE训练配置 (普通图像模式)")
    print("=" * 50)
    
    # GPU检测
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_properties(0).name
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("❌ 未检测到CUDA")
        return
    
    # 根据GPU选择配置
    if "P100" in gpu_name or gpu_memory > 14:
        config = {
            "batch_size": 16,
            "mixed_precision": "no",
            "learning_rate": "0.0001",
            "gradient_accumulation": 1,
        }
    elif "T4" in gpu_name or gpu_memory > 10:
        config = {
            "batch_size": 12,
            "mixed_precision": "fp16", 
            "learning_rate": "0.0001",
            "gradient_accumulation": 2,
        }
    else:
        config = {
            "batch_size": 8,
            "mixed_precision": "fp16",
            "learning_rate": "0.0001", 
            "gradient_accumulation": 2,
        }
    
    print(f"\n📊 训练参数:")
    print(f"   批次大小: {config['batch_size']}")
    print(f"   梯度累积: {config['gradient_accumulation']}")
    print(f"   有效批次: {config['batch_size'] * config['gradient_accumulation']}")
    print(f"   学习率: {config['learning_rate']}")
    print(f"   混合精度: {config['mixed_precision']}")
    print(f"   训练轮数: 80")
    
    print(f"\n⚖️  损失权重:")
    print(f"   重建损失: 1.0 (MSE)")
    print(f"   KL散度: 1e-6 (极低，避免过度正则化)")
    print(f"   感知损失: 0.1 (LPIPS)")
    print(f"   频域损失: 0.0 (禁用)")
    
    print(f"\n🏗️  模型架构:")
    print(f"   输入分辨率: 64×64×3")
    print(f"   下采样层: 3层 (64→32→16→8)")
    print(f"   通道数: [64, 128, 256]")
    print(f"   潜在维度: 8×8×4")
    print(f"   压缩比: 48:1")
    
    print(f"\n🖼️  训练模式:")
    print(f"   ✅ 普通图像训练")
    print(f"   ✅ 标准MSE重建损失")
    print(f"   ✅ 轻量感知损失")
    print(f"   ❌ 频域损失 (已禁用)")
    print(f"   ❌ 微多普勒特殊处理 (已禁用)")
    
    print(f"\n🎯 预期效果:")
    print(f"   目标PSNR: >25dB")
    print(f"   训练时间: ~10-15分钟")
    print(f"   显存使用: ~3-4GB")
    print(f"   训练稳定性: 高")
    
    print(f"\n💡 优势:")
    print(f"   ✅ 简化损失函数，减少复杂性")
    print(f"   ✅ 更稳定的训练过程")
    print(f"   ✅ 避免频域损失的潜在问题")
    print(f"   ✅ 遵循标准图像VAE训练流程")
    print(f"   ✅ 更好的调试和问题定位")

if __name__ == "__main__":
    show_current_config()
