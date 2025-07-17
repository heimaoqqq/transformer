#!/usr/bin/env python3
"""
追踪VAE每一层的输出尺寸变化
验证下采样的具体过程
"""

import torch
from diffusers import AutoencoderKL
import torch.nn as nn

class LayerTracker:
    """追踪每一层的输出"""
    def __init__(self):
        self.layer_outputs = []
        self.layer_names = []
    
    def hook_fn(self, name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                self.layer_outputs.append(output.shape)
                self.layer_names.append(name)
                print(f"   {name}: {input[0].shape} → {output.shape}")
            elif hasattr(output, 'sample'):  # DiagonalGaussianDistribution
                self.layer_outputs.append(output.sample().shape)
                self.layer_names.append(name)
                print(f"   {name}: {input[0].shape} → {output.sample().shape}")
        return hook

def trace_vae_architecture():
    """详细追踪VAE架构的每一层"""
    print("🔍 详细追踪VAE架构 - 每一层的尺寸变化")
    print("=" * 70)
    
    device = "cpu"  # 使用CPU避免CUDA问题
    
    # 创建3层配置的VAE (我们修复后的配置)
    print("\n🏗️ 创建3层DownEncoderBlock2D配置:")
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
        up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
        block_out_channels=[128, 256, 512],
        latent_channels=4,
        sample_size=128,
        layers_per_block=1,
        act_fn="silu",
        norm_num_groups=32,
        scaling_factor=0.18215,
    ).to(device)
    
    # 创建追踪器
    tracker = LayerTracker()
    
    # 注册钩子函数来追踪编码器的每一层
    print("\n📊 编码器层级结构:")
    print("   编码器输入层:")
    vae.encoder.conv_in.register_forward_hook(tracker.hook_fn("conv_in"))
    
    print("   下采样块:")
    for i, down_block in enumerate(vae.encoder.down_blocks):
        down_block.register_forward_hook(tracker.hook_fn(f"down_block_{i}"))
    
    print("   中间层:")
    vae.encoder.mid_block.register_forward_hook(tracker.hook_fn("mid_block"))
    
    print("   输出层:")
    vae.encoder.conv_norm_out.register_forward_hook(tracker.hook_fn("conv_norm_out"))
    vae.encoder.conv_out.register_forward_hook(tracker.hook_fn("conv_out"))
    
    # 测试输入
    test_input = torch.randn(1, 3, 128, 128).to(device)
    print(f"\n🎯 输入图像: {test_input.shape}")
    print("\n📈 编码过程 (逐层追踪):")
    
    with torch.no_grad():
        # 编码
        posterior = vae.encode(test_input).latent_dist
        latent = posterior.sample()
        
        print(f"\n✅ 最终潜在空间: {latent.shape}")
        
        # 解码
        print(f"\n📉 解码过程:")
        reconstructed = vae.decode(latent).sample
        print(f"✅ 重建输出: {reconstructed.shape}")
    
    # 分析结果
    print(f"\n🎯 分析结果:")
    print(f"   输入: 128×128 = 16,384 像素")
    print(f"   潜在: 32×32 = 1,024 像素")
    print(f"   空间压缩比: {128//32}×{128//32} = {(128//32)**2}倍")
    print(f"   总压缩比: {(128*128*3)//(32*32*4)}:1")

def analyze_downsampling_pattern():
    """分析下采样模式"""
    print("\n" + "="*70)
    print("🔬 分析下采样模式")
    
    # 测试不同层数的配置
    configs = [
        (1, [128], "1层"),
        (2, [128, 256], "2层"), 
        (3, [128, 256, 512], "3层"),
        (4, [128, 256, 512, 512], "4层")
    ]
    
    device = "cpu"
    
    for num_layers, channels, name in configs:
        print(f"\n📊 {name}配置:")
        try:
            down_blocks = ["DownEncoderBlock2D"] * num_layers
            up_blocks = ["UpDecoderBlock2D"] * num_layers
            
            vae = AutoencoderKL(
                in_channels=3,
                out_channels=3,
                down_block_types=down_blocks,
                up_block_types=up_blocks,
                block_out_channels=channels,
                latent_channels=4,
                sample_size=128,
                layers_per_block=1,
                act_fn="silu",
                norm_num_groups=32,
                scaling_factor=0.18215,
            ).to(device)
            
            test_input = torch.randn(1, 3, 128, 128).to(device)
            with torch.no_grad():
                latent = vae.encode(test_input).latent_dist.sample()
            
            actual_factor = 128 // latent.shape[-1]
            theoretical_factor = 2 ** num_layers
            
            print(f"   输入: {test_input.shape}")
            print(f"   潜在: {latent.shape}")
            print(f"   实际下采样: {actual_factor}倍 (128→{latent.shape[-1]})")
            print(f"   理论下采样: {theoretical_factor}倍")
            print(f"   规律: 实际 = 理论 ÷ 2 (第一层不下采样)")
            
        except Exception as e:
            print(f"   ❌ {name}配置失败: {e}")

if __name__ == "__main__":
    trace_vae_architecture()
    analyze_downsampling_pattern()
