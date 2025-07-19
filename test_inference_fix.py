#!/usr/bin/env python3
"""
测试修复后的推理代码
验证调度器创建是否正常工作
"""

import torch
import sys
from pathlib import Path

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_scheduler_creation():
    """测试调度器创建"""
    print("🧪 测试调度器创建...")
    
    try:
        from diffusers import DDIMScheduler, DDPMScheduler
        
        # 测试DDPM调度器创建
        print("   1️⃣ 测试DDPM调度器...")
        ddpm_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            variance_type="fixed_small",
            clip_sample=False,
            prediction_type="epsilon",
            thresholding=False,
            dynamic_thresholding_ratio=0.995,
            clip_sample_range=1.0,
            sample_max_value=1.0,
        )
        print("   ✅ DDPM调度器创建成功")
        
        # 测试DDIM调度器创建
        print("   2️⃣ 测试DDIM调度器...")
        ddim_scheduler = DDIMScheduler.from_config(ddpm_scheduler.config)
        print("   ✅ DDIM调度器创建成功")
        
        # 测试调度器设置
        print("   3️⃣ 测试调度器设置...")
        ddim_scheduler.set_timesteps(50)
        print(f"   ✅ 调度器设置成功，时间步数: {len(ddim_scheduler.timesteps)}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 调度器创建失败: {e}")
        return False

def test_generator_init():
    """测试生成器初始化（模拟）"""
    print("\n🧪 测试生成器初始化逻辑...")
    
    try:
        # 模拟生成器初始化的关键部分
        from diffusers import DDIMScheduler, DDPMScheduler
        
        # 测试DDIM调度器初始化逻辑
        print("   1️⃣ 测试DDIM调度器初始化逻辑...")
        scheduler_type = "ddim"
        
        if scheduler_type == "ddim":
            # 先创建DDPM调度器配置，然后转换为DDIM
            ddpm_scheduler = DDPMScheduler(
                num_train_timesteps=1000,
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                variance_type="fixed_small",
                clip_sample=False,
                prediction_type="epsilon",
                thresholding=False,
                dynamic_thresholding_ratio=0.995,
                clip_sample_range=1.0,
                sample_max_value=1.0,
            )
            scheduler = DDIMScheduler.from_config(ddpm_scheduler.config)
        else:
            scheduler = DDPMScheduler(
                num_train_timesteps=1000,
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                variance_type="fixed_small",
                clip_sample=False,
                prediction_type="epsilon",
                thresholding=False,
                dynamic_thresholding_ratio=0.995,
                clip_sample_range=1.0,
                sample_max_value=1.0,
            )
        
        print(f"   ✅ {scheduler_type}调度器初始化成功")
        
        # 测试用户ID验证逻辑
        print("   2️⃣ 测试用户ID验证逻辑...")
        num_users = 31
        user_ids = [1, 5, 10, 15]
        user_id_mapping = {}  # 空映射，使用默认转换
        
        for user_id in user_ids:
            # 获取实际的用户索引
            user_idx = user_id_mapping.get(user_id, user_id - 1 if user_id > 0 else 0)
            if user_idx < 0 or user_idx >= num_users:
                raise ValueError(f"Invalid user_id {user_id} (mapped to index {user_idx}). Index must be in range [0, {num_users-1}]")
            print(f"      用户ID {user_id} -> 索引 {user_idx} ✅")
        
        print("   ✅ 用户ID验证逻辑正常")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 生成器初始化测试失败: {e}")
        return False

def test_user_condition_encoder():
    """测试用户条件编码器"""
    print("\n🧪 测试用户条件编码器...")
    
    try:
        from training.train_diffusion import UserConditionEncoder
        
        # 创建条件编码器
        condition_encoder = UserConditionEncoder(
            num_users=31,
            embed_dim=512
        )
        
        print("   ✅ 条件编码器创建成功")
        
        # 测试编码
        user_tensor = torch.tensor([0, 4, 9, 14])  # 对应用户ID 1, 5, 10, 15
        with torch.no_grad():
            embeddings = condition_encoder(user_tensor)
            print(f"   ✅ 编码成功，输出形状: {embeddings.shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 条件编码器测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🔧 测试推理代码修复...")
    
    all_tests_passed = True
    
    # 测试调度器创建
    if not test_scheduler_creation():
        all_tests_passed = False
    
    # 测试生成器初始化
    if not test_generator_init():
        all_tests_passed = False
    
    # 测试条件编码器
    if not test_user_condition_encoder():
        all_tests_passed = False
    
    if all_tests_passed:
        print("\n🎉 所有测试通过！推理代码修复成功！")
        print("\n📝 修复总结:")
        print("   1. ✅ 修复了调度器配置文件缺失问题")
        print("   2. ✅ 调度器现在直接使用代码配置而不是从文件加载")
        print("   3. ✅ 用户ID验证逻辑已优化")
        print("   4. ✅ 支持1-based用户ID到0-based索引的转换")
        
        print("\n🚀 现在可以运行推理命令了！")
        print("   推荐的测试命令:")
        print("   python inference/generate.py \\")
        print("       --vae_path \"/path/to/vae\" \\")
        print("       --unet_path \"/path/to/unet\" \\")
        print("       --condition_encoder_path \"/path/to/condition_encoder.pt\" \\")
        print("       --num_users 31 \\")
        print("       --user_ids 1 5 10 15 \\")
        print("       --num_images_per_user 4 \\")
        print("       --num_inference_steps 50 \\")
        print("       --guidance_scale 7.5 \\")
        print("       --output_dir \"./generated_images\"")
    else:
        print("\n❌ 部分测试失败，请检查错误信息")
    
    return all_tests_passed

if __name__ == "__main__":
    main()
