#!/usr/bin/env python3
"""
基于训练时推理逻辑的生成脚本
完全复制训练时的generate_samples函数逻辑
"""

import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from typing import List
from tqdm import tqdm

from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, DDPMScheduler
import sys

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from training.train_diffusion import UserConditionEncoder

def generate_images_training_style(
    vae_path: str,
    unet_path: str,
    condition_encoder_path: str,
    user_ids: List[int],
    num_users: int,
    num_images_per_user: int = 1,
    num_inference_steps: int = 20,
    output_dir: str = "./generated_images",
    device: str = "auto",
    seed: int = 42,
    data_dir: str = None  # 新增：用于获取正确的用户映射
):
    """
    使用训练时的逻辑生成图像
    完全复制train_diffusion.py中的generate_samples函数

    Args:
        data_dir: 训练数据目录，用于获取正确的用户ID映射
    """
    
    # 设备检测
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 使用设备: {device}")
    
    # 设置随机种子
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # 获取正确的用户ID映射 (修复关键问题)
    user_id_to_idx = {}
    if data_dir is not None:
        print("🔍 获取训练时的用户ID映射...")
        from pathlib import Path
        data_path = Path(data_dir)
        all_users = []

        # 扫描数据目录，获取所有用户ID (与训练时逻辑一致)
        for user_dir in data_path.iterdir():
            if user_dir.is_dir() and user_dir.name.startswith('ID_'):
                try:
                    user_id = int(user_dir.name.split('_')[1])
                    all_users.append(user_id)
                except ValueError:
                    continue

        # 排序并创建映射 (与训练时逻辑一致)
        all_users = sorted(all_users)
        user_id_to_idx = {user_id: idx for idx, user_id in enumerate(all_users)}

        print(f"  找到 {len(all_users)} 个用户: {all_users}")
        print(f"  用户ID映射: {user_id_to_idx}")
    else:
        print("⚠️  未提供数据目录，使用简单映射 (可能不正确)")
        # 回退到简单映射
        for user_id in user_ids:
            user_id_to_idx[user_id] = user_id - 1 if user_id > 0 else user_id
    
    # 1. 加载VAE (与训练时相同的方式)
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(vae_path)
    vae.to(device)
    vae.eval()
    
    # 2. 加载UNet (与训练时相同的方式)
    print("Loading UNet...")
    unet = UNet2DConditionModel.from_pretrained(unet_path)
    unet.to(device)
    unet.eval()
    
    print(f"UNet配置:")
    print(f"  - cross_attention_dim: {unet.config.cross_attention_dim}")
    print(f"  - in_channels: {unet.config.in_channels}")
    print(f"  - sample_size: {unet.config.sample_size}")
    
    # 3. 创建条件编码器 (与训练时相同的方式)
    print("Creating Condition Encoder...")
    condition_encoder = UserConditionEncoder(
        num_users=num_users,
        embed_dim=unet.config.cross_attention_dim  # 使用UNet的cross_attention_dim
    )
    
    # 4. 加载条件编码器权重
    print("Loading Condition Encoder weights...")
    if Path(condition_encoder_path).is_dir():
        condition_encoder_file = Path(condition_encoder_path) / "condition_encoder.pt"
    else:
        condition_encoder_file = Path(condition_encoder_path)
    
    if condition_encoder_file.exists():
        try:
            condition_encoder.load_state_dict(torch.load(condition_encoder_file, map_location=device))
            print("✅ 成功加载条件编码器权重")
        except Exception as e:
            print(f"⚠️  加载条件编码器权重失败: {e}")
            print("   将使用随机初始化权重")
    else:
        print(f"⚠️  条件编码器文件不存在: {condition_encoder_file}")
        print("   将使用随机初始化权重")
    
    condition_encoder.to(device)
    condition_encoder.eval()
    
    # 5. 创建噪声调度器 (与训练时相同)
    print("Creating noise scheduler...")
    noise_scheduler = DDPMScheduler(
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
    
    # 6. 创建DDIM调度器用于推理 (与训练时相同)
    ddim_scheduler = DDIMScheduler.from_config(noise_scheduler.config)
    ddim_scheduler.set_timesteps(num_inference_steps)
    
    print(f"✅ 所有模型加载完成，开始生成...")
    
    # 7. 生成图像 (完全复制训练时的逻辑)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for user_id in user_ids:
            print(f"Generating {num_images_per_user} images for user {user_id}...")
            
            user_dir = output_path / f"user_{user_id:02d}"
            user_dir.mkdir(exist_ok=True)
            
            for img_idx in range(num_images_per_user):
                # 随机噪声 (与训练时完全相同)
                latents = torch.randn(1, 4, 32, 32, device=device)
                
                # 用户条件编码 (使用正确的映射)
                if user_id in user_id_to_idx:
                    user_idx = user_id_to_idx[user_id]
                    print(f"  用户 {user_id} → 索引 {user_idx}")
                else:
                    print(f"  ⚠️  用户 {user_id} 不在映射中，使用默认索引 0")
                    user_idx = 0

                user_idx_tensor = torch.tensor([user_idx], device=device)
                encoder_hidden_states = condition_encoder(user_idx_tensor)
                encoder_hidden_states = encoder_hidden_states.unsqueeze(1)
                
                # 去噪过程 (与训练时完全相同)
                for t in tqdm(ddim_scheduler.timesteps, desc=f"User {user_id}, Image {img_idx+1}"):
                    timestep = t.unsqueeze(0).to(device)
                    
                    # 预测噪声
                    noise_pred = unet(
                        latents,
                        timestep,
                        encoder_hidden_states=encoder_hidden_states,
                        return_dict=False
                    )[0]
                    
                    # 去噪步骤
                    latents = ddim_scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                
                # VAE解码 (与训练时完全相同)
                vae_model = vae.module if hasattr(vae, 'module') else vae
                latents = latents / vae_model.config.scaling_factor
                image = vae_model.decode(latents).sample
                
                # 转换为PIL图像 (与训练时完全相同)
                image = image.clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
                image = (image * 255).astype(np.uint8)
                
                # 保存图像
                pil_image = Image.fromarray(image)
                pil_image.save(user_dir / f"generated_{img_idx:03d}.png")
                
                # 清理内存
                del latents
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    print(f"✅ 生成完成！图像保存到: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate images using training-style logic")
    
    # 模型路径
    parser.add_argument("--vae_path", type=str, required=True, help="VAE模型路径")
    parser.add_argument("--unet_path", type=str, required=True, help="UNet模型路径")
    parser.add_argument("--condition_encoder_path", type=str, required=True, help="条件编码器路径")
    parser.add_argument("--num_users", type=int, required=True, help="用户总数")
    parser.add_argument("--data_dir", type=str, help="训练数据目录 (用于获取正确的用户ID映射)")
    
    # 生成参数
    parser.add_argument("--user_ids", type=int, nargs="+", required=True, help="要生成的用户ID列表")
    parser.add_argument("--num_images_per_user", type=int, default=5, help="每个用户生成的图像数量")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="推理步数")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default="./generated_images", help="输出目录")
    parser.add_argument("--device", type=str, default="auto", help="设备")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    generate_images_training_style(
        vae_path=args.vae_path,
        unet_path=args.unet_path,
        condition_encoder_path=args.condition_encoder_path,
        user_ids=args.user_ids,
        num_users=args.num_users,
        num_images_per_user=args.num_images_per_user,
        num_inference_steps=args.num_inference_steps,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        data_dir=args.data_dir
    )

if __name__ == "__main__":
    main()
