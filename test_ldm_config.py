#!/usr/bin/env python3
"""
LDM(扩散模型)训练配置测试脚本
验证扩散模型架构配置是否正确
"""

import torch
import warnings
warnings.filterwarnings("ignore")

def test_ldm_config():
    """测试LDM配置"""
    print("🔍 测试LDM(扩散模型)训练配置")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📱 设备: {device}")
    
    # 测试UNet配置
    print(f"\n🎯 测试UNet配置:")
    try:
        from diffusers import UNet2DConditionModel
        
        # 使用修复后的配置
        unet = UNet2DConditionModel(
            sample_size=32,  # 修复: 直接设置为实际潜在尺寸
            in_channels=4,   # VAE潜在维度
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
            attention_head_dim=8,
            use_linear_projection=True,
            class_embed_type=None,
            num_class_embeds=None,
        ).to(device)
        
        total_params = sum(p.numel() for p in unet.parameters())
        print(f"   ✅ UNet创建成功 - 参数量: {total_params:,}")
        print(f"   📐 sample_size: 32 (匹配VAE潜在空间)")
        print(f"   🔗 cross_attention_dim: 768")
        print(f"   🧱 layers_per_block: 2")
        
        # 测试前向传播
        with torch.no_grad():
            # 模拟输入
            latents = torch.randn(2, 4, 32, 32).to(device)
            timesteps = torch.randint(0, 1000, (2,)).to(device)
            encoder_hidden_states = torch.randn(2, 1, 768).to(device)
            
            # UNet前向传播
            noise_pred = unet(
                latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False
            )[0]
            
            print(f"   ✅ 前向传播成功:")
            print(f"      输入潜在: {latents.shape}")
            print(f"      时间步: {timesteps.shape}")
            print(f"      条件嵌入: {encoder_hidden_states.shape}")
            print(f"      噪声预测: {noise_pred.shape}")
            
            if noise_pred.shape == latents.shape:
                print(f"   ✅ 输出形状正确")
            else:
                print(f"   ❌ 输出形状错误")
                return False
        
    except Exception as e:
        print(f"   ❌ UNet配置测试失败: {e}")
        return False
    
    # 测试调度器配置
    print(f"\n⏰ 测试调度器配置:")
    try:
        from diffusers import DDPMScheduler, DDIMScheduler
        
        # DDPM调度器
        ddpm_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            variance_type="fixed_small",
            clip_sample=False,
            prediction_type="epsilon",
        )
        
        print(f"   ✅ DDPM调度器创建成功")
        print(f"      训练时间步: {ddpm_scheduler.config.num_train_timesteps}")
        print(f"      beta范围: {ddpm_scheduler.config.beta_start} - {ddpm_scheduler.config.beta_end}")
        print(f"      预测类型: {ddpm_scheduler.config.prediction_type}")
        
        # DDIM调度器 (用于推理)
        ddim_scheduler = DDIMScheduler.from_config(ddpm_scheduler.config)
        ddim_scheduler.set_timesteps(50)
        
        print(f"   ✅ DDIM调度器创建成功")
        print(f"      推理步数: 50")
        
        # 测试调度器功能
        with torch.no_grad():
            test_latents = torch.randn(1, 4, 32, 32)
            test_noise = torch.randn_like(test_latents)
            test_timesteps = torch.randint(0, 1000, (1,))
            
            # 添加噪声
            noisy_latents = ddpm_scheduler.add_noise(test_latents, test_noise, test_timesteps)
            print(f"   ✅ 噪声添加测试通过: {noisy_latents.shape}")
            
            # 去噪步骤
            step_result = ddim_scheduler.step(test_noise, test_timesteps[0], test_latents)
            print(f"   ✅ 去噪步骤测试通过: {step_result.prev_sample.shape}")
        
    except Exception as e:
        print(f"   ❌ 调度器配置测试失败: {e}")
        return False
    
    # 测试条件编码器
    print(f"\n🎭 测试条件编码器:")
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent / "training"))
        
        from train_diffusion import UserConditionEncoder
        
        # 创建条件编码器
        num_users = 31  # 假设31个用户
        condition_encoder = UserConditionEncoder(
            num_users=num_users,
            embed_dim=768,
            dropout=0.1
        ).to(device)
        
        total_params = sum(p.numel() for p in condition_encoder.parameters())
        print(f"   ✅ 条件编码器创建成功 - 参数量: {total_params:,}")
        print(f"   👥 用户数量: {num_users}")
        print(f"   📏 嵌入维度: 768")
        
        # 测试条件编码
        with torch.no_grad():
            user_ids = torch.tensor([0, 5, 10]).to(device)
            user_embeds = condition_encoder(user_ids)
            
            print(f"   ✅ 条件编码测试通过:")
            print(f"      输入用户ID: {user_ids.shape}")
            print(f"      输出嵌入: {user_embeds.shape}")
            
            if user_embeds.shape == (3, 768):
                print(f"   ✅ 嵌入形状正确")
            else:
                print(f"   ❌ 嵌入形状错误")
                return False
        
    except Exception as e:
        print(f"   ❌ 条件编码器测试失败: {e}")
        return False
    
    # 测试完整训练流程
    print(f"\n🔄 测试完整训练流程:")
    try:
        with torch.no_grad():
            # 模拟训练数据
            batch_size = 2
            latents = torch.randn(batch_size, 4, 32, 32).to(device)
            user_ids = torch.tensor([1, 15]).to(device)
            
            # 编码条件
            encoder_hidden_states = condition_encoder(user_ids)
            encoder_hidden_states = encoder_hidden_states.unsqueeze(1)  # [B, 1, embed_dim]
            
            # 添加噪声
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
            noisy_latents = ddpm_scheduler.add_noise(latents, noise, timesteps)
            
            # UNet预测
            model_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False
            )[0]
            
            # 计算损失
            loss = torch.nn.functional.mse_loss(model_pred, noise)
            
            print(f"   ✅ 完整训练流程测试通过:")
            print(f"      批次大小: {batch_size}")
            print(f"      潜在空间: {latents.shape}")
            print(f"      条件嵌入: {encoder_hidden_states.shape}")
            print(f"      噪声预测: {model_pred.shape}")
            print(f"      训练损失: {loss.item():.6f}")
        
    except Exception as e:
        print(f"   ❌ 完整训练流程测试失败: {e}")
        return False
    
    # 显示训练配置
    print(f"\n📋 LDM训练配置总结:")
    config = {
        "resolution": 128,
        "latent_size": 32,
        "unet_sample_size": 32,
        "cross_attention_dim": 768,
        "num_train_timesteps": 1000,
        "batch_size": 4,
        "learning_rate": 1e-4,
        "num_epochs": 100,
        "num_users": 31
    }
    
    for key, value in config.items():
        print(f"   ✅ {key}: {value}")
    
    return True

def show_training_command():
    """显示正确的LDM训练命令"""
    print(f"\n🚀 正确的LDM训练命令:")
    print(f"python training/train_diffusion.py \\")
    print(f"    --data_dir \"/kaggle/input/dataset\" \\")
    print(f"    --vae_path \"/kaggle/working/outputs/vae/final_model\" \\")
    print(f"    --resolution 128 \\")
    print(f"    --batch_size 4 \\")
    print(f"    --num_epochs 100 \\")
    print(f"    --learning_rate 1e-4 \\")
    print(f"    --cross_attention_dim 768 \\")
    print(f"    --output_dir \"/kaggle/working/outputs/diffusion\"")

def main():
    """主函数"""
    success = test_ldm_config()
    
    if success:
        show_training_command()
        print(f"\n✅ LDM配置测试通过！可以开始训练。")
    else:
        print(f"\n❌ LDM配置测试失败，请检查配置。")

if __name__ == "__main__":
    main()
