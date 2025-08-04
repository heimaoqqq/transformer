#!/usr/bin/env python3
"""
支持指导强度的条件扩散图像生成脚本
专门针对256×256微多普勒时频图像优化

新增功能:
- 支持分类器自由指导 (CFG)
- 可配置指导强度 (guidance_scale)
- 针对256×256图像优化
- 自动检测数据集格式
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
sys.path.insert(0, str(current_dir))

# 直接定义UserConditionEncoder类，避免导入问题
import torch.nn as nn

class UserConditionEncoder(nn.Module):
    """用户条件编码器 - 完全匹配训练时的结构"""
    def __init__(self, num_users: int, embed_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.num_users = num_users
        self.embed_dim = embed_dim

        # 用户嵌入层
        self.user_embedding = nn.Embedding(num_users, embed_dim)

        # MLP层 - 完全匹配训练代码结构
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),  # mlp.0
            nn.SiLU(),                        # mlp.1 (激活函数)
            nn.Dropout(dropout),              # mlp.2 (Dropout)
            nn.Linear(embed_dim, embed_dim),  # mlp.3
        )

        # 初始化 - 匹配训练代码
        nn.init.normal_(self.user_embedding.weight, std=0.02)

    def forward(self, user_indices):
        """
        编码用户ID
        Args:
            user_indices: 用户索引 [B]
        Returns:
            用户嵌入 [B, embed_dim]
        """
        # 获取用户嵌入
        user_embeds = self.user_embedding(user_indices)

        # 通过MLP
        user_embeds = self.mlp(user_embeds)

        return user_embeds

def generate_with_guidance(
    vae_path: str,
    unet_path: str,
    condition_encoder_path: str,
    user_ids: List[int],
    data_dir: str,
    num_images_per_user: int = 50,
    num_inference_steps: int = 50,
    guidance_scale: float = 1.0,  # 新增: 指导强度
    output_dir: str = "./generated_images",
    device: str = "auto",
    seed: int = 42
):
    """
    使用指导强度生成条件图像
    
    Args:
        guidance_scale: 指导强度
            - 1.0: 纯条件生成 (与训练时相同)
            - >1.0: 分类器自由指导 (CFG), 增强条件控制
            - 推荐值: 1.0-3.0 (过高可能导致过饱和)
    """
    
    # 设备检测
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 使用设备: {device}")
    print(f"🎯 指导强度: {guidance_scale} ({'纯条件' if guidance_scale == 1.0 else 'CFG增强'})")
    
    # 设置随机种子
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # 获取用户ID映射
    print("🔍 扫描数据集...")
    data_path = Path(data_dir)
    user_labels = []  # 收集所有用户标签（与训练时一致）

    for user_dir in data_path.iterdir():
        if user_dir.is_dir() and user_dir.name.startswith('ID_'):
            try:
                user_id = int(user_dir.name.split('_')[1])
                image_files = list(user_dir.glob("*.png")) + list(user_dir.glob("*.jpg"))
                if len(image_files) > 0:
                    # 为每个图像添加用户标签（模拟训练时的数据加载）
                    user_labels.extend([user_id] * len(image_files))
                    print(f"  用户 {user_id:2d}: {len(image_files):3d} 张图像")
            except ValueError:
                continue

    # 与训练时完全一致的映射逻辑
    unique_users = sorted(list(set(user_labels)))
    user_id_to_idx = {user: idx for idx, user in enumerate(unique_users)}
    num_users = len(unique_users)
    
    print(f"📊 发现 {num_users} 个用户: {unique_users}")
    print(f"🗺️  用户映射: {user_id_to_idx}")
    print(f"🔧 修复: 使用与训练时一致的映射逻辑")
    
    # 验证目标用户
    for user_id in user_ids:
        if user_id not in user_id_to_idx:
            print(f"❌ 用户 {user_id} 不在数据集中")
            return False
    
    # 加载模型
    print("\n📦 加载模型...")
    
    # 1. VAE
    print("  加载VAE...")
    vae = AutoencoderKL.from_pretrained(vae_path)
    vae.to(device)
    vae.eval()
    
    # 2. UNet
    print("  加载UNet...")
    unet = UNet2DConditionModel.from_pretrained(unet_path)
    unet.to(device)
    unet.eval()
    
    # 3. 条件编码器
    print("  加载条件编码器...")
    condition_encoder = UserConditionEncoder(
        num_users=num_users,
        embed_dim=unet.config.cross_attention_dim
    )
    condition_encoder_state = torch.load(condition_encoder_path, map_location='cpu')
    condition_encoder.load_state_dict(condition_encoder_state)
    condition_encoder.to(device)
    condition_encoder.eval()
    
    # 4. 调度器
    print("  创建调度器...")
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        prediction_type="epsilon",
    )
    ddim_scheduler = DDIMScheduler.from_config(noise_scheduler.config)
    ddim_scheduler.set_timesteps(num_inference_steps)
    
    print(f"✅ 模型加载完成")
    
    # 生成图像
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🎨 开始生成图像...")
    print(f"  输出目录: {output_path}")
    print(f"  推理步数: {num_inference_steps}")
    print(f"  每用户图像数: {num_images_per_user}")
    
    with torch.no_grad():
        for user_id in user_ids:
            print(f"\n👤 生成用户 {user_id} 的图像...")
            
            user_dir = output_path / f"user_{user_id:02d}"
            user_dir.mkdir(exist_ok=True)
            
            user_idx = user_id_to_idx[user_id]
            print(f"  用户 {user_id} → 索引 {user_idx}")
            
            for img_idx in range(num_images_per_user):
                # 初始噪声 (256×256 → 32×32潜在空间)
                latents = torch.randn(1, 4, 32, 32, device=device)
                
                # 用户条件编码
                user_idx_tensor = torch.tensor([user_idx], device=device)
                encoder_hidden_states = condition_encoder(user_idx_tensor)
                encoder_hidden_states = encoder_hidden_states.unsqueeze(1)
                
                # 去噪过程
                for t in tqdm(ddim_scheduler.timesteps, 
                            desc=f"用户 {user_id}, 图像 {img_idx+1}/{num_images_per_user}",
                            leave=False):
                    timestep = t.unsqueeze(0).to(device)
                    
                    if guidance_scale == 1.0:
                        # 纯条件生成 (与训练时相同)
                        noise_pred = unet(
                            latents,
                            timestep,
                            encoder_hidden_states=encoder_hidden_states,
                            return_dict=False
                        )[0]
                    else:
                        # 分类器自由指导 (CFG)
                        # 条件预测
                        noise_pred_cond = unet(
                            latents,
                            timestep,
                            encoder_hidden_states=encoder_hidden_states,
                            return_dict=False
                        )[0]
                        
                        # 无条件预测 (空条件)
                        uncond_embeddings = torch.zeros_like(encoder_hidden_states)
                        noise_pred_uncond = unet(
                            latents,
                            timestep,
                            encoder_hidden_states=uncond_embeddings,
                            return_dict=False
                        )[0]
                        
                        # CFG组合
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    
                    # 去噪步骤
                    latents = ddim_scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                
                # VAE解码
                vae_model = vae.module if hasattr(vae, 'module') else vae
                latents = latents / vae_model.config.scaling_factor
                image = vae_model.decode(latents).sample
                
                # 转换为PIL图像
                image = image.clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
                image = (image * 255).astype(np.uint8)
                
                # 保存图像
                pil_image = Image.fromarray(image)
                
                # 添加生成信息标签
                try:
                    from PIL import ImageDraw, ImageFont
                    draw = ImageDraw.Draw(pil_image)
                    label_text = f"ID:{user_id} CFG:{guidance_scale} Steps:{num_inference_steps}"
                    draw.text((5, 5), label_text, fill=(255, 255, 255))
                    draw.text((5, 20), f"256x256 Generated", fill=(255, 255, 255))
                except:
                    pass
                
                pil_image.save(user_dir / f"generated_{img_idx:03d}.png")
                
                # 清理内存
                del latents
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            print(f"  ✅ 用户 {user_id} 完成: {user_dir}")
    
    print(f"\n🎉 生成完成！")
    print(f"📁 输出目录: {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="支持指导强度的条件扩散图像生成")
    
    # 模型路径
    parser.add_argument("--vae_path", type=str, required=True, help="VAE模型路径")
    parser.add_argument("--unet_path", type=str, required=True, help="UNet模型路径")
    parser.add_argument("--condition_encoder_path", type=str, required=True, help="条件编码器路径")
    parser.add_argument("--data_dir", type=str, required=True, help="数据目录 (用于获取用户映射)")
    
    # 生成参数
    parser.add_argument("--user_ids", type=int, nargs="+", required=True, help="要生成的用户ID列表")
    parser.add_argument("--num_images_per_user", type=int, default=50, help="每用户生成图像数")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="DDIM推理步数")
    parser.add_argument("--guidance_scale", type=float, default=1.0, 
                       help="指导强度 (1.0=纯条件, >1.0=CFG增强, 推荐1.0-3.0)")
    parser.add_argument("--output_dir", type=str, default="./generated_images", help="输出目录")
    parser.add_argument("--device", type=str, default="auto", help="计算设备")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    print("🎨 支持指导强度的条件扩散图像生成")
    print("=" * 60)
    print(f"📊 数据目录: {args.data_dir}")
    print(f"🎯 目标用户: {args.user_ids}")
    print(f"🎛️  指导强度: {args.guidance_scale}")
    print(f"📈 推理步数: {args.num_inference_steps}")
    print(f"📁 输出目录: {args.output_dir}")
    
    success = generate_with_guidance(
        vae_path=args.vae_path,
        unet_path=args.unet_path,
        condition_encoder_path=args.condition_encoder_path,
        user_ids=args.user_ids,
        data_dir=args.data_dir,
        num_images_per_user=args.num_images_per_user,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed
    )
    
    if success:
        print("\n🎉 生成成功完成!")
        return 0
    else:
        print("\n❌ 生成失败!")
        return 1

if __name__ == "__main__":
    exit(main())
