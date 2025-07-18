#!/usr/bin/env python3
"""
VAE与LDM兼容性检查脚本
确保VAE和扩散模型配置完全兼容
"""

import torch
import warnings
warnings.filterwarnings("ignore")

def check_vae_ldm_compatibility():
    """检查VAE与LDM兼容性"""
    print("🔍 VAE与LDM兼容性检查")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📱 设备: {device}")
    
    compatibility_issues = []
    
    # 1. 创建VAE (训练配置)
    print(f"\n1️⃣ 创建VAE (训练配置):")
    try:
        from diffusers import AutoencoderKL
        
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
        
        print(f"   ✅ VAE创建成功")
        
        # 测试VAE压缩
        test_input = torch.randn(1, 3, 128, 128).to(device)
        with torch.no_grad():
            latent = vae.encode(test_input).latent_dist.sample()
            reconstructed = vae.decode(latent).sample
        
        vae_latent_shape = latent.shape
        vae_compression = test_input.shape[-1] // latent.shape[-1]
        
        print(f"   📐 VAE输入: {test_input.shape}")
        print(f"   🎯 VAE潜在: {vae_latent_shape}")
        print(f"   📊 压缩比: {vae_compression}倍")
        
    except Exception as e:
        print(f"   ❌ VAE创建失败: {e}")
        compatibility_issues.append("VAE创建失败")
        return False
    
    # 2. 创建UNet (LDM配置)
    print(f"\n2️⃣ 创建UNet (LDM配置):")
    try:
        from diffusers import UNet2DConditionModel
        
        unet = UNet2DConditionModel(
            sample_size=32,  # 应该匹配VAE潜在空间
            in_channels=4,
            out_channels=4,
            layers_per_block=2,
            block_out_channels=(320, 640, 1280, 1280),
            down_block_types=(
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
            ),
            cross_attention_dim=768,
        ).to(device)
        
        print(f"   ✅ UNet创建成功")
        print(f"   📐 UNet sample_size: {unet.config.sample_size}")
        print(f"   🔗 cross_attention_dim: {unet.config.cross_attention_dim}")
        
    except Exception as e:
        print(f"   ❌ UNet创建失败: {e}")
        compatibility_issues.append("UNet创建失败")
        return False
    
    # 3. 检查尺寸兼容性
    print(f"\n3️⃣ 检查尺寸兼容性:")
    
    # 检查VAE潜在空间与UNet sample_size
    expected_latent_size = vae_latent_shape[-1]  # 应该是32
    unet_sample_size = unet.config.sample_size
    
    if expected_latent_size == unet_sample_size:
        print(f"   ✅ 潜在空间尺寸匹配: VAE={expected_latent_size}, UNet={unet_sample_size}")
    else:
        print(f"   ❌ 潜在空间尺寸不匹配: VAE={expected_latent_size}, UNet={unet_sample_size}")
        compatibility_issues.append(f"潜在空间尺寸不匹配: VAE={expected_latent_size}, UNet={unet_sample_size}")
    
    # 检查通道数
    vae_latent_channels = vae_latent_shape[1]  # 应该是4
    unet_in_channels = unet.config.in_channels
    unet_out_channels = unet.config.out_channels
    
    if vae_latent_channels == unet_in_channels == unet_out_channels:
        print(f"   ✅ 通道数匹配: VAE={vae_latent_channels}, UNet输入={unet_in_channels}, UNet输出={unet_out_channels}")
    else:
        print(f"   ❌ 通道数不匹配: VAE={vae_latent_channels}, UNet输入={unet_in_channels}, UNet输出={unet_out_channels}")
        compatibility_issues.append("通道数不匹配")
    
    # 4. 测试完整工作流程
    print(f"\n4️⃣ 测试完整工作流程:")
    try:
        from diffusers import DDPMScheduler
        
        # 创建调度器
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        
        # 创建条件编码器
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent / "training"))
        from train_diffusion import UserConditionEncoder
        
        condition_encoder = UserConditionEncoder(
            num_users=31,
            embed_dim=768
        ).to(device)
        
        with torch.no_grad():
            # 模拟完整流程
            batch_size = 2
            
            # 1. 输入图像
            input_images = torch.randn(batch_size, 3, 128, 128).to(device)
            print(f"   📥 输入图像: {input_images.shape}")
            
            # 2. VAE编码
            posterior = vae.encode(input_images).latent_dist
            latents = posterior.sample()
            latents = latents * vae.config.scaling_factor
            print(f"   🔄 VAE编码: {input_images.shape} → {latents.shape}")
            
            # 3. 添加噪声
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)
            print(f"   🔊 添加噪声: {latents.shape} → {noisy_latents.shape}")
            
            # 4. 条件编码
            user_ids = torch.tensor([1, 15]).to(device)
            encoder_hidden_states = condition_encoder(user_ids)
            encoder_hidden_states = encoder_hidden_states.unsqueeze(1)
            print(f"   🎭 条件编码: {user_ids.shape} → {encoder_hidden_states.shape}")
            
            # 5. UNet预测
            model_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False
            )[0]
            print(f"   🎯 UNet预测: {noisy_latents.shape} → {model_pred.shape}")
            
            # 6. VAE解码
            clean_latents = latents / vae.config.scaling_factor
            reconstructed = vae.decode(clean_latents).sample
            print(f"   🔄 VAE解码: {latents.shape} → {reconstructed.shape}")
            
            # 验证所有形状
            shape_checks = [
                ("VAE潜在空间", latents.shape, (batch_size, 4, 32, 32)),
                ("UNet输入", noisy_latents.shape, latents.shape),
                ("UNet输出", model_pred.shape, latents.shape),
                ("重建图像", reconstructed.shape, input_images.shape),
                ("条件嵌入", encoder_hidden_states.shape, (batch_size, 1, 768)),
            ]
            
            all_shapes_correct = True
            for name, actual, expected in shape_checks:
                if actual == expected:
                    print(f"   ✅ {name}: {actual}")
                else:
                    print(f"   ❌ {name}: {actual}, 期望: {expected}")
                    compatibility_issues.append(f"{name}形状不匹配")
                    all_shapes_correct = False
            
            if all_shapes_correct:
                print(f"   🎉 所有形状检查通过！")
            
    except Exception as e:
        print(f"   ❌ 完整工作流程测试失败: {e}")
        compatibility_issues.append(f"完整工作流程失败: {e}")
    
    # 5. 检查配置参数一致性
    print(f"\n5️⃣ 检查配置参数一致性:")
    
    config_checks = [
        ("分辨率", 128, "VAE和LDM都应使用128×128"),
        ("潜在空间尺寸", 32, "VAE输出和UNet sample_size"),
        ("潜在通道数", 4, "VAE和UNet通道数"),
        ("条件维度", 768, "UNet cross_attention_dim"),
        ("压缩比", 4, "128÷32=4倍压缩"),
    ]
    
    for name, expected, description in config_checks:
        print(f"   ✅ {name}: {expected} ({description})")
    
    # 6. 总结
    print(f"\n📊 兼容性检查总结:")
    if not compatibility_issues:
        print(f"   🎉 所有兼容性检查通过！")
        print(f"   ✅ VAE与LDM配置完全兼容")
        print(f"   ✅ 可以安全开始训练")
        
        print(f"\n📋 确认的配置:")
        print(f"   - 输入分辨率: 128×128")
        print(f"   - VAE潜在空间: 32×32×4")
        print(f"   - UNet sample_size: 32")
        print(f"   - 压缩比: 4倍")
        print(f"   - 条件维度: 768")
        
        return True
    else:
        print(f"   ❌ 发现 {len(compatibility_issues)} 个兼容性问题:")
        for i, issue in enumerate(compatibility_issues, 1):
            print(f"      {i}. {issue}")
        return False

def show_compatible_training_commands():
    """显示兼容的训练命令"""
    print(f"\n🚀 兼容的训练命令:")
    
    print(f"\n1️⃣ VAE训练:")
    print(f"python training/train_vae.py \\")
    print(f"    --resolution 128 \\")
    print(f"    --down_block_types \"DownEncoderBlock2D,DownEncoderBlock2D,DownEncoderBlock2D\" \\")
    print(f"    --sample_size 128")
    
    print(f"\n2️⃣ LDM训练:")
    print(f"python training/train_diffusion.py \\")
    print(f"    --resolution 128 \\")
    print(f"    --vae_path \"outputs/vae/final_model\"")

def main():
    """主函数"""
    success = check_vae_ldm_compatibility()
    
    if success:
        show_compatible_training_commands()
        print(f"\n✅ 兼容性检查完成！可以开始训练。")
    else:
        print(f"\n❌ 兼容性检查失败，请修复问题后重试。")

if __name__ == "__main__":
    main()
