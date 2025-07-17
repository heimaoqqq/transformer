#!/usr/bin/env python3
"""
微多普勒时频图条件生成脚本
基于训练好的VAE和条件扩散模型生成指定用户的图像
"""

import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
import json
from typing import List, Optional, Union
from tqdm import tqdm

from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, DDPMScheduler
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from training.train_diffusion import UserConditionEncoder

class MicroDopplerGenerator:
    """微多普勒图像生成器"""
    
    def __init__(
        self,
        vae_path: str,
        unet_path: str,
        condition_encoder_path: str,
        num_users: int,
        device: str = "cuda",
        scheduler_type: str = "ddim"
    ):
        """
        初始化生成器
        
        Args:
            vae_path: VAE模型路径
            unet_path: UNet模型路径  
            condition_encoder_path: 条件编码器路径
            num_users: 用户总数
            device: 设备
            scheduler_type: 调度器类型 ("ddim" 或 "ddpm")
        """
        self.device = device
        self.num_users = num_users
        
        # 加载VAE
        print("Loading VAE...")
        self.vae = AutoencoderKL.from_pretrained(vae_path)
        self.vae.to(device)
        self.vae.eval()
        
        # 加载UNet
        print("Loading UNet...")
        self.unet = UNet2DConditionModel.from_pretrained(unet_path)
        self.unet.to(device)
        self.unet.eval()
        
        # 加载条件编码器
        print("Loading Condition Encoder...")
        self.condition_encoder = UserConditionEncoder(
            num_users=num_users,
            embed_dim=self.unet.config.cross_attention_dim
        )
        self.condition_encoder.load_state_dict(torch.load(condition_encoder_path, map_location=device))
        self.condition_encoder.to(device)
        self.condition_encoder.eval()
        
        # 创建调度器
        if scheduler_type == "ddim":
            self.scheduler = DDIMScheduler.from_pretrained(unet_path, subfolder="scheduler")
        else:
            self.scheduler = DDPMScheduler.from_pretrained(unet_path, subfolder="scheduler")
        
        print(f"Generator initialized with {scheduler_type} scheduler")
    
    @torch.no_grad()
    def generate(
        self,
        user_ids: Union[int, List[int]],
        num_images_per_user: int = 1,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None,
        return_latents: bool = False
    ) -> List[Image.Image]:
        """
        生成微多普勒图像
        
        Args:
            user_ids: 用户ID或用户ID列表
            num_images_per_user: 每个用户生成的图像数量
            num_inference_steps: 推理步数
            guidance_scale: 引导强度
            generator: 随机数生成器
            return_latents: 是否返回潜在表示
            
        Returns:
            生成的图像列表
        """
        # 处理用户ID
        if isinstance(user_ids, int):
            user_ids = [user_ids]
        
        # 验证用户ID
        for user_id in user_ids:
            if user_id < 0 or user_id >= self.num_users:
                raise ValueError(f"Invalid user_id {user_id}. Must be in range [0, {self.num_users-1}]")
        
        # 设置调度器
        self.scheduler.set_timesteps(num_inference_steps)
        
        generated_images = []
        
        for user_id in user_ids:
            print(f"Generating {num_images_per_user} images for user {user_id}...")
            
            for i in range(num_images_per_user):
                # 初始化随机噪声
                latents = torch.randn(
                    (1, self.unet.config.in_channels, 32, 32),  # 256/8 = 32
                    generator=generator,
                    device=self.device,
                    dtype=self.unet.dtype
                )
                
                # 缩放初始噪声
                latents = latents * self.scheduler.init_noise_sigma
                
                # 编码用户条件
                user_tensor = torch.tensor([user_id], device=self.device)
                encoder_hidden_states = self.condition_encoder(user_tensor)
                encoder_hidden_states = encoder_hidden_states.unsqueeze(1)  # [1, 1, embed_dim]
                
                # 无条件嵌入 (用于classifier-free guidance)
                if guidance_scale > 1.0:
                    uncond_user_tensor = torch.tensor([0], device=self.device)  # 假设0是无条件token
                    uncond_encoder_hidden_states = self.condition_encoder(uncond_user_tensor)
                    uncond_encoder_hidden_states = uncond_encoder_hidden_states.unsqueeze(1)
                    
                    # 拼接条件和无条件嵌入
                    encoder_hidden_states = torch.cat([
                        uncond_encoder_hidden_states, encoder_hidden_states
                    ])
                    
                    # 复制latents用于classifier-free guidance
                    latents = torch.cat([latents] * 2)
                
                # 去噪循环
                for t in tqdm(self.scheduler.timesteps, desc=f"User {user_id}, Image {i+1}"):
                    # 扩展latents用于classifier-free guidance
                    latent_model_input = self.scheduler.scale_model_input(latents, t)
                    
                    # 预测噪声
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=encoder_hidden_states,
                        return_dict=False
                    )[0]
                    
                    # 执行classifier-free guidance
                    if guidance_scale > 1.0:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                        
                        # 只保留条件部分的latents
                        latents = latents.chunk(2)[1]
                    
                    # 计算前一个噪声样本
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                
                # 如果需要返回潜在表示
                if return_latents:
                    generated_images.append(latents)
                else:
                    # VAE解码
                    latents = latents / self.vae.config.scaling_factor
                    image = self.vae.decode(latents).sample
                    
                    # 转换为PIL图像
                    image = (image / 2 + 0.5).clamp(0, 1)
                    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
                    image = (image * 255).astype(np.uint8)
                    generated_images.append(Image.fromarray(image))
        
        return generated_images
    
    def generate_interpolation(
        self,
        user_id1: int,
        user_id2: int,
        num_steps: int = 10,
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None
    ) -> List[Image.Image]:
        """
        在两个用户之间生成插值图像
        
        Args:
            user_id1: 起始用户ID
            user_id2: 结束用户ID
            num_steps: 插值步数
            num_inference_steps: 推理步数
            generator: 随机数生成器
            
        Returns:
            插值图像列表
        """
        # 获取用户嵌入
        user1_tensor = torch.tensor([user_id1], device=self.device)
        user2_tensor = torch.tensor([user_id2], device=self.device)
        
        user1_embed = self.condition_encoder(user1_tensor)
        user2_embed = self.condition_encoder(user2_tensor)
        
        # 设置调度器
        self.scheduler.set_timesteps(num_inference_steps)
        
        interpolated_images = []
        
        for i in range(num_steps):
            # 计算插值权重
            alpha = i / (num_steps - 1)
            
            # 插值用户嵌入
            interpolated_embed = (1 - alpha) * user1_embed + alpha * user2_embed
            interpolated_embed = interpolated_embed.unsqueeze(1)
            
            # 生成图像
            latents = torch.randn(
                (1, self.unet.config.in_channels, 32, 32),
                generator=generator,
                device=self.device,
                dtype=self.unet.dtype
            )
            latents = latents * self.scheduler.init_noise_sigma
            
            # 去噪循环
            for t in self.scheduler.timesteps:
                latent_model_input = self.scheduler.scale_model_input(latents, t)
                
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=interpolated_embed,
                    return_dict=False
                )[0]
                
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            
            # VAE解码
            latents = latents / self.vae.config.scaling_factor
            image = self.vae.decode(latents).sample
            
            # 转换为PIL图像
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
            interpolated_images.append(Image.fromarray(image))
        
        return interpolated_images

def main():
    parser = argparse.ArgumentParser(description="Generate Micro-Doppler Images")
    
    # 模型路径
    parser.add_argument("--vae_path", type=str, required=True, help="VAE模型路径")
    parser.add_argument("--unet_path", type=str, required=True, help="UNet模型路径")
    parser.add_argument("--condition_encoder_path", type=str, required=True, help="条件编码器路径")
    parser.add_argument("--num_users", type=int, required=True, help="用户总数")
    
    # 生成参数
    parser.add_argument("--user_ids", type=int, nargs="+", required=True, help="要生成的用户ID列表")
    parser.add_argument("--num_images_per_user", type=int, default=5, help="每个用户生成的图像数量")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="推理步数")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="引导强度")
    parser.add_argument("--scheduler_type", type=str, default="ddim", choices=["ddim", "ddpm"], help="调度器类型")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default="./generated_images", help="输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    # 特殊功能
    parser.add_argument("--interpolation", action="store_true", help="生成插值图像")
    parser.add_argument("--interpolation_users", type=int, nargs=2, help="插值的两个用户ID")
    parser.add_argument("--interpolation_steps", type=int, default=10, help="插值步数")
    
    args = parser.parse_args()
    
    # 设置随机种子
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        generator = torch.Generator().manual_seed(args.seed)
    else:
        generator = None
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化生成器
    generator_model = MicroDopplerGenerator(
        vae_path=args.vae_path,
        unet_path=args.unet_path,
        condition_encoder_path=args.condition_encoder_path,
        num_users=args.num_users,
        scheduler_type=args.scheduler_type
    )
    
    if args.interpolation and args.interpolation_users:
        # 生成插值图像
        print(f"Generating interpolation between users {args.interpolation_users[0]} and {args.interpolation_users[1]}")
        
        images = generator_model.generate_interpolation(
            user_id1=args.interpolation_users[0],
            user_id2=args.interpolation_users[1],
            num_steps=args.interpolation_steps,
            num_inference_steps=args.num_inference_steps,
            generator=generator
        )
        
        # 保存插值图像
        interp_dir = output_dir / f"interpolation_{args.interpolation_users[0]}_{args.interpolation_users[1]}"
        interp_dir.mkdir(exist_ok=True)
        
        for i, img in enumerate(images):
            img.save(interp_dir / f"step_{i:03d}.png")
        
        # 创建拼接图像
        width, height = images[0].size
        combined = Image.new('RGB', (width * len(images), height))
        for i, img in enumerate(images):
            combined.paste(img, (i * width, 0))
        combined.save(interp_dir / "combined.png")
        
        print(f"Interpolation images saved to {interp_dir}")
    
    else:
        # 生成指定用户的图像
        print(f"Generating images for users: {args.user_ids}")
        
        images = generator_model.generate(
            user_ids=args.user_ids,
            num_images_per_user=args.num_images_per_user,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator
        )
        
        # 保存图像
        img_idx = 0
        for user_id in args.user_ids:
            user_dir = output_dir / f"user_{user_id:02d}"
            user_dir.mkdir(exist_ok=True)
            
            for i in range(args.num_images_per_user):
                images[img_idx].save(user_dir / f"generated_{i:03d}.png")
                img_idx += 1
        
        print(f"Generated images saved to {output_dir}")

if __name__ == "__main__":
    main()
