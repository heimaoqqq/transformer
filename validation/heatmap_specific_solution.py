#!/usr/bin/env python3
"""
针对热力图数据的专门解决方案
基于用户展示的ID_1和ID_2热力图，提供针对性的验证策略
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

def analyze_heatmap_characteristics():
    """分析热力图数据的特点"""
    print("🔥 热力图数据特征分析")
    print("=" * 50)
    
    print("📊 观察到的特征:")
    print("  1. 都是蓝-绿-黄-红的热力图")
    print("  2. 主要差异在细微的模式变化")
    print("  3. 颜色分布和整体结构很相似")
    print("  4. 差异主要体现在局部细节")
    
    print("\n🚨 这种数据的挑战:")
    print("  1. 用户间差异极小，肉眼都难以区分")
    print("  2. 扩散模型倾向于生成'平均'样本")
    print("  3. 微小差异容易被噪声掩盖")
    print("  4. 需要极强的条件控制才能保持差异")
    
    return True

def create_extreme_guidance_config():
    """创建极端指导强度配置"""
    print("\n🚀 针对热力图数据的极端配置")
    print("=" * 50)
    
    configs = {
        "conservative": {
            "guidance_scale": 20.0,
            "num_inference_steps": 100,
            "condition_dropout": 0.05,
            "description": "保守方案 - 适合初次尝试"
        },
        "aggressive": {
            "guidance_scale": 35.0,
            "num_inference_steps": 150,
            "condition_dropout": 0.02,
            "description": "激进方案 - 强化用户特征"
        },
        "extreme": {
            "guidance_scale": 50.0,
            "num_inference_steps": 200,
            "condition_dropout": 0.01,
            "description": "极端方案 - 最大化条件控制"
        }
    }
    
    for name, config in configs.items():
        print(f"\n📋 {config['description']} ({name}):")
        print(f"  指导强度: {config['guidance_scale']}")
        print(f"  推理步数: {config['num_inference_steps']}")
        print(f"  条件dropout: {config['condition_dropout']}")
    
    return configs

def generate_with_extreme_guidance(
    vae_path: str,
    unet_path: str, 
    condition_encoder_path: str,
    data_dir: str,
    user_id: int,
    guidance_scale: float = 35.0,
    num_inference_steps: int = 150,
    num_images: int = 8,
    output_dir: str = "extreme_guidance_test"
):
    """使用极端指导强度生成图像"""
    print(f"\n🎯 使用极端指导强度生成用户 {user_id} 的图像")
    print(f"指导强度: {guidance_scale}")
    print(f"推理步数: {num_inference_steps}")
    
    try:
        # 导入必要的模块
        from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
        from training.train_diffusion import UserConditionEncoder, create_user_id_mapping
        
        # 设备设置
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")
        
        # 加载模型
        print("📂 加载模型...")
        vae = AutoencoderKL.from_pretrained(vae_path)
        unet = UNet2DConditionModel.from_pretrained(unet_path)
        
        # 创建用户ID映射
        user_id_mapping = create_user_id_mapping(data_dir)
        num_users = len(user_id_mapping)
        
        # 加载条件编码器
        condition_encoder = UserConditionEncoder(
            num_users=num_users,
            embed_dim=unet.config.cross_attention_dim
        )
        
        try:
            condition_encoder.load_state_dict(torch.load(condition_encoder_path, map_location='cpu'))
            print("✅ 成功加载条件编码器权重")
        except Exception as e:
            print(f"⚠️  使用随机权重: {e}")
        
        # 移动到设备
        vae = vae.to(device)
        unet = unet.to(device)
        condition_encoder = condition_encoder.to(device)
        
        # 创建调度器
        scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        scheduler.set_timesteps(num_inference_steps)
        
        # 准备生成
        if user_id not in user_id_mapping:
            print(f"❌ 用户 {user_id} 不在映射中: {list(user_id_mapping.keys())}")
            return False
        
        user_idx = user_id_mapping[user_id]
        print(f"用户 {user_id} 映射到索引 {user_idx}")
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 生成图像
        print(f"🎨 开始生成 {num_images} 张图像...")
        
        with torch.no_grad():
            for i in range(num_images):
                print(f"  生成第 {i+1}/{num_images} 张...")
                
                # 随机噪声
                latents = torch.randn(1, 4, 32, 32, device=device)
                
                # 用户条件
                user_tensor = torch.tensor([user_idx], device=device)
                user_embedding = condition_encoder(user_tensor)
                
                # 扩散过程
                latents = latents * scheduler.init_noise_sigma
                
                for t in scheduler.timesteps:
                    # 有条件预测
                    noise_pred_cond = unet(
                        latents, 
                        t, 
                        encoder_hidden_states=user_embedding
                    ).sample
                    
                    # 无条件预测（使用零嵌入）
                    zero_embedding = torch.zeros_like(user_embedding)
                    noise_pred_uncond = unet(
                        latents,
                        t,
                        encoder_hidden_states=zero_embedding
                    ).sample
                    
                    # 分类器自由指导 - 使用极高的指导强度
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    
                    # 调度器步骤
                    latents = scheduler.step(noise_pred, t, latents).prev_sample
                
                # 解码为图像
                latents = 1 / 0.18215 * latents
                images = vae.decode(latents).sample
                images = (images / 2 + 0.5).clamp(0, 1)
                
                # 保存图像
                from torchvision.utils import save_image
                save_path = output_path / f"user_{user_id}_extreme_guidance_{i+1}.png"
                save_image(images, save_path)
                
                print(f"    保存到: {save_path}")
        
        print(f"✅ 生成完成，图像保存在: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_extreme_guidance_results(
    generated_dir: str,
    real_data_root: str,
    user_id: int,
    classifier_path: str = None
):
    """验证极端指导强度的结果"""
    print(f"\n🔍 验证极端指导强度的生成结果")
    
    # 如果没有分类器，先训练一个
    if not classifier_path or not Path(classifier_path).exists():
        print("📚 训练分类器...")
        from validation.improved_single_user_validation import train_user_classifier
        
        classifier_path = f"extreme_guidance_classifier_user_{user_id}.pth"
        success = train_user_classifier(
            target_user_id=user_id,
            real_data_root=real_data_root,
            output_dir="extreme_guidance_validation",
            max_samples_per_class=500,  # 使用更多数据
            epochs=30,                  # 更多训练轮数
            batch_size=16
        )
        
        if not success:
            print("❌ 分类器训练失败")
            return False
    
    # 验证生成图像
    print("🎯 验证生成图像...")
    from validation.improved_single_user_validation import validate_generated_images
    
    results = validate_generated_images(
        generated_images_dir=generated_dir,
        classifier_path=classifier_path,
        target_user_id=user_id
    )
    
    return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="热力图数据专门解决方案")
    parser.add_argument("--action", choices=["analyze", "generate", "validate"], required=True)
    parser.add_argument("--user_id", type=int, default=1)
    parser.add_argument("--data_dir", type=str, help="数据目录")
    parser.add_argument("--vae_path", type=str, help="VAE路径")
    parser.add_argument("--unet_path", type=str, help="UNet路径")
    parser.add_argument("--condition_encoder_path", type=str, help="条件编码器路径")
    parser.add_argument("--guidance_scale", type=float, default=35.0, help="指导强度")
    parser.add_argument("--num_inference_steps", type=int, default=150, help="推理步数")
    parser.add_argument("--num_images", type=int, default=8, help="生成图像数量")
    parser.add_argument("--output_dir", type=str, default="extreme_guidance_test")
    
    args = parser.parse_args()
    
    if args.action == "analyze":
        analyze_heatmap_characteristics()
        create_extreme_guidance_config()
        
    elif args.action == "generate":
        if not all([args.vae_path, args.unet_path, args.condition_encoder_path, args.data_dir]):
            print("❌ 生成需要提供所有模型路径和数据目录")
            return
        
        success = generate_with_extreme_guidance(
            vae_path=args.vae_path,
            unet_path=args.unet_path,
            condition_encoder_path=args.condition_encoder_path,
            data_dir=args.data_dir,
            user_id=args.user_id,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            num_images=args.num_images,
            output_dir=args.output_dir
        )
        
        if success:
            print(f"\n🎯 下一步: 验证生成结果")
            print(f"python validation/heatmap_specific_solution.py --action validate --user_id {args.user_id} --data_dir {args.data_dir}")
    
    elif args.action == "validate":
        if not all([args.data_dir]):
            print("❌ 验证需要提供数据目录")
            return
        
        validate_extreme_guidance_results(
            generated_dir=args.output_dir,
            real_data_root=args.data_dir,
            user_id=args.user_id
        )

if __name__ == "__main__":
    main()
