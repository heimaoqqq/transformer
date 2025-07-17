#!/usr/bin/env python3
"""
调试VAE架构 - 找出正确的配置
"""

import torch
from diffusers import AutoencoderKL

def test_vae_configurations():
    """测试不同的VAE配置"""
    print("🔍 调试VAE架构配置")
    print("=" * 50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 测试不同的配置
    configs = [
        {
            "name": "2层配置",
            "down_blocks": ["DownEncoderBlock2D", "DownEncoderBlock2D"],
            "up_blocks": ["UpDecoderBlock2D", "UpDecoderBlock2D"],
            "channels": [128, 256],
            "expected": "128→64→32"
        },
        {
            "name": "3层配置", 
            "down_blocks": ["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
            "up_blocks": ["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
            "channels": [128, 256, 256],
            "expected": "128→64→32→16"
        },
        {
            "name": "Stable Diffusion标准",
            "down_blocks": ["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
            "up_blocks": ["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
            "channels": [128, 256, 512, 512],
            "expected": "128→64→32→16→8"
        }
    ]
    
    test_input = torch.randn(1, 3, 128, 128).to(device)
    
    for config in configs:
        print(f"\n🧪 测试 {config['name']}:")
        print(f"   期望: {config['expected']}")
        
        try:
            vae = AutoencoderKL(
                in_channels=3,
                out_channels=3,
                down_block_types=config["down_blocks"],
                up_block_types=config["up_blocks"],
                block_out_channels=config["channels"],
                latent_channels=4,
                sample_size=128,
                layers_per_block=2,
                act_fn="silu",
                norm_num_groups=32,
                scaling_factor=0.18215,
            ).to(device)
            
            with torch.no_grad():
                latent = vae.encode(test_input).latent_dist.sample()
                reconstructed = vae.decode(latent).sample
                
            print(f"   ✅ 潜在空间: {latent.shape}")
            print(f"   ✅ 重建形状: {reconstructed.shape}")
            
            # 计算实际的下采样倍数
            h_ratio = 128 / latent.shape[2]
            w_ratio = 128 / latent.shape[3]
            print(f"   📊 下采样倍数: {h_ratio}x{w_ratio}")
            
            # 计算压缩比
            input_size = 128 * 128 * 3
            latent_size = latent.shape[1] * latent.shape[2] * latent.shape[3]
            compression_ratio = input_size / latent_size
            print(f"   📊 压缩比: {compression_ratio:.1f}:1")
            
            # 检查是否符合我们的目标 (32×32)
            if latent.shape[2] == 32 and latent.shape[3] == 32:
                print(f"   🎯 ✅ 符合目标 32×32!")
                return config
            else:
                print(f"   ⚠️  不符合目标 32×32")
                
        except Exception as e:
            print(f"   ❌ 配置失败: {e}")
    
    return None

def find_correct_config_for_32x32():
    """找到正确的32×32配置"""
    print(f"\n🎯 寻找32×32的正确配置...")
    
    # 分析: 128 → 32 需要4倍下采样
    # 每层DownEncoderBlock2D通常下采样2倍
    # 所以需要: 128 → 64 → 32 (2层)
    
    # 但如果实际测试显示2层得到64×64，说明每层只下采样√2倍
    # 或者有其他机制
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_input = torch.randn(1, 3, 128, 128).to(device)
    
    # 尝试不同的layers_per_block
    for layers_per_block in [1, 2]:
        print(f"\n🔧 测试 layers_per_block={layers_per_block}:")
        
        try:
            vae = AutoencoderKL(
                in_channels=3,
                out_channels=3,
                down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
                up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
                block_out_channels=[128, 256],
                latent_channels=4,
                sample_size=128,
                layers_per_block=layers_per_block,
                act_fn="silu",
                norm_num_groups=32,
                scaling_factor=0.18215,
            ).to(device)
            
            with torch.no_grad():
                latent = vae.encode(test_input).latent_dist.sample()
                
            print(f"   潜在空间: {latent.shape}")
            
            if latent.shape[2] == 32:
                print(f"   🎯 找到了! layers_per_block={layers_per_block}")
                return {"layers_per_block": layers_per_block}
                
        except Exception as e:
            print(f"   ❌ 失败: {e}")
    
    # 如果还是不行，尝试调整sample_size
    print(f"\n🔧 测试调整sample_size:")
    for sample_size in [64, 128, 256]:
        try:
            vae = AutoencoderKL(
                in_channels=3,
                out_channels=3,
                down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
                up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
                block_out_channels=[128, 256],
                latent_channels=4,
                sample_size=sample_size,
                layers_per_block=2,
                act_fn="silu",
                norm_num_groups=32,
                scaling_factor=0.18215,
            ).to(device)
            
            with torch.no_grad():
                latent = vae.encode(test_input).latent_dist.sample()
                
            print(f"   sample_size={sample_size}: 潜在空间 {latent.shape}")
            
            if latent.shape[2] == 32:
                print(f"   🎯 找到了! sample_size={sample_size}")
                return {"sample_size": sample_size}
                
        except Exception as e:
            print(f"   ❌ sample_size={sample_size} 失败: {e}")
    
    return None

def main():
    """主函数"""
    print("🔍 VAE架构调试工具")
    
    # 测试不同配置
    best_config = test_vae_configurations()
    
    if not best_config:
        # 如果没找到，尝试其他方法
        correction = find_correct_config_for_32x32()
        if correction:
            print(f"\n💡 建议修正: {correction}")
    
    print(f"\n📝 总结:")
    print(f"   目标: 128×128 → 32×32×4")
    print(f"   问题: 当前配置产生64×64而不是32×32")
    print(f"   解决: 需要调整架构参数")

if __name__ == "__main__":
    main()
