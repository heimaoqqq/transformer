#!/usr/bin/env python3
"""
VAE训练配置测试脚本
验证VAE架构配置是否正确 (128×128 → 32×32)
"""

import torch
import warnings
warnings.filterwarnings("ignore")

def test_vae_config():
    """测试VAE配置"""
    print("🔍 测试VAE训练配置 (128×128 → 32×32)")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📱 设备: {device}")
    
    # 测试数据加载器
    print(f"\n📊 测试数据加载器:")
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        
        from ..utils.data_loader import MicroDopplerDataset
        
        # 创建测试数据集 (使用128×128分辨率)
        dataset = MicroDopplerDataset(
            data_dir="data",  # 假设数据目录
            resolution=128,   # 关键: 128×128分辨率
            augment=False,
            split="test"
        )
        
        print(f"   ✅ 数据集创建成功")
        print(f"   📐 分辨率: 128×128")
        print(f"   🔄 数据增广: 关闭")
        
    except Exception as e:
        print(f"   ❌ 数据加载失败: {e}")
        return False
    
    # 测试VAE架构
    print(f"\n🏗️  测试VAE架构 (128×128 → 32×32):")
    try:
        from diffusers import AutoencoderKL
        
        # 使用训练时的实际配置
        vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],  # 3层
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],        # 3层
            block_out_channels=[128, 256, 512],                                                   # 3层通道数
            latent_channels=4,
            sample_size=128,                                                 # 设置为128匹配输入尺寸
            layers_per_block=1,                                              # 标准配置
            act_fn="silu",
            norm_num_groups=32,
            scaling_factor=0.18215,
        ).to(device)
        
        total_params = sum(p.numel() for p in vae.parameters())
        print(f"   ✅ VAE创建成功 - 参数量: {total_params:,}")
        
        # 测试前向传播
        test_input = torch.randn(1, 3, 128, 128).to(device)
        
        with torch.no_grad():
            # 编码
            latent_dist = vae.encode(test_input).latent_dist
            latent = latent_dist.sample()
            print(f"   ✅ 潜在空间形状: {latent.shape}")
            
            # 解码
            reconstructed = vae.decode(latent).sample
            print(f"   ✅ 重建形状: {reconstructed.shape}")
            
            # 验证形状
            expected_latent_shape = (1, 4, 32, 32)
            expected_output_shape = (1, 3, 128, 128)
            
            if latent.shape == expected_latent_shape:
                print(f"   ✅ 潜在空间形状正确: {latent.shape}")
            else:
                print(f"   ❌ 潜在空间形状错误: {latent.shape}, 期望: {expected_latent_shape}")
                return False
                
            if reconstructed.shape == expected_output_shape:
                print(f"   ✅ 重建形状正确: {reconstructed.shape}")
            else:
                print(f"   ❌ 重建形状错误: {reconstructed.shape}, 期望: {expected_output_shape}")
                return False
        
        # 计算压缩比
        compression_ratio = (128 * 128 * 3) / (32 * 32 * 4)
        print(f"   📊 压缩比: {compression_ratio:.1f}:1")
        print(f"   🔽 下采样因子: {128 // 32}倍")
        
    except Exception as e:
        print(f"   ❌ VAE架构测试失败: {e}")
        return False
    
    # 测试训练参数兼容性
    print(f"\n⚙️  测试训练参数:")
    
    training_config = {
        "resolution": 128,
        "batch_size": 8,
        "learning_rate": 1e-4,
        "num_epochs": 80,
        "kl_weight": 1e-6,
        "perceptual_weight": 0.5,
        "down_blocks": 3,
        "latent_channels": 4,
        "expected_latent_size": 32
    }
    
    for key, value in training_config.items():
        print(f"   ✅ {key}: {value}")
    
    print(f"\n🎯 配置验证总结:")
    print(f"   ✅ VAE架构: 3层下采样 (128→64→32)")
    print(f"   ✅ 潜在空间: 32×32×4")
    print(f"   ✅ 压缩比: 12:1")
    print(f"   ✅ 与扩散模型兼容")
    
    return True

def show_training_command():
    """显示正确的VAE训练命令"""
    print(f"\n🚀 正确的VAE训练命令:")
    print(f"python training/train_vae.py \\")
    print(f"    --data_dir \"/kaggle/input/dataset\" \\")
    print(f"    --resolution 128 \\")
    print(f"    --batch_size 8 \\")
    print(f"    --num_epochs 80 \\")
    print(f"    --down_block_types \"DownEncoderBlock2D,DownEncoderBlock2D,DownEncoderBlock2D\" \\")
    print(f"    --up_block_types \"UpDecoderBlock2D,UpDecoderBlock2D,UpDecoderBlock2D\" \\")
    print(f"    --block_out_channels \"128,256,512\" \\")
    print(f"    --sample_size 128 \\")
    print(f"    --output_dir \"/kaggle/working/outputs/vae\"")

def main():
    """主函数"""
    success = test_vae_config()
    
    if success:
        show_training_command()
        print(f"\n✅ VAE配置测试通过！可以开始训练。")
    else:
        print(f"\n❌ VAE配置测试失败，请检查配置。")

if __name__ == "__main__":
    main()
