#!/usr/bin/env python3
"""
微多普勒时频图条件扩散模型训练脚本
基于 Diffusers UNet2DConditionModel 实现
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import (
    AutoencoderKL, 
    UNet2DConditionModel, 
    DDPMScheduler,
    DDIMScheduler
)
from accelerate import Accelerator
from tqdm import tqdm
import wandb
from pathlib import Path
import argparse
import json
import numpy as np
from PIL import Image

# 导入自定义模块
import sys
sys.path.append('..')
from utils.data_loader import MicroDopplerDataset, MicroDopplerDataModule

class UserConditionEncoder(nn.Module):
    """用户ID条件编码器"""
    
    def __init__(self, num_users: int, embed_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.num_users = num_users
        self.embed_dim = embed_dim
        
        # 用户嵌入层
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        
        # 可选的MLP层
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        
        # 初始化
        nn.init.normal_(self.user_embedding.weight, std=0.02)
    
    def forward(self, user_indices: torch.Tensor) -> torch.Tensor:
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

def train_diffusion(args):
    """条件扩散模型训练主函数"""
    
    # 初始化加速器
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb" if args.use_wandb else None,
        project_dir=args.output_dir
    )
    
    # 设置随机种子
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化wandb
    if args.use_wandb and accelerator.is_main_process:
        wandb.init(
            project="micro-doppler-diffusion",
            config=vars(args),
            name=f"diffusion-{args.experiment_name}"
        )
    
    # 加载预训练的VAE
    vae = AutoencoderKL.from_pretrained(args.vae_path)
    vae.requires_grad_(False)  # 冻结VAE参数
    vae.eval()
    
    # 创建数据模块
    data_module = MicroDopplerDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        resolution=args.resolution,
        val_split=args.val_split,
        seed=args.seed
    )
    
    # 获取数据加载器
    train_dataloader = data_module.get_dataloader("train")
    val_dataloader = data_module.get_dataloader("val")
    
    # 获取用户数量
    num_users = len(data_module.all_users)
    print(f"Training with {num_users} users")
    
    # 创建用户条件编码器
    condition_encoder = UserConditionEncoder(
        num_users=num_users,
        embed_dim=args.cross_attention_dim,
        dropout=args.condition_dropout
    )
    
    # 创建UNet模型
    unet = UNet2DConditionModel(
        sample_size=args.resolution // 8,  # VAE压缩8倍
        in_channels=4,  # VAE潜在维度
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
        cross_attention_dim=args.cross_attention_dim,
        attention_head_dim=8,
        use_linear_projection=True,
        class_embed_type=None,
        num_class_embeds=None,
        upcast_attention=False,
        resnet_time_scale_shift="default",
        time_embedding_type="positional",
        time_embedding_dim=None,
        time_embedding_act_fn=None,
        timestep_post_act=None,
        time_cond_proj_dim=None,
    )
    
    # 创建噪声调度器
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        trained_betas=None,
        variance_type="fixed_small",
        clip_sample=False,
        prediction_type="epsilon",
        thresholding=False,
        dynamic_thresholding_ratio=0.995,
        clip_sample_range=1.0,
        sample_max_value=1.0,
    )
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        list(unet.parameters()) + list(condition_encoder.parameters()),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8
    )
    
    # 创建学习率调度器
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.learning_rate * 0.1
    )
    
    # 使用accelerator准备模型和数据
    unet, condition_encoder, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        unet, condition_encoder, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )
    
    # VAE移到正确的设备
    vae = vae.to(accelerator.device)
    
    # 训练循环
    global_step = 0
    
    for epoch in range(args.num_epochs):
        unet.train()
        condition_encoder.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1}/{args.num_epochs}",
            disable=not accelerator.is_local_main_process
        )
        
        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(unet):
                # 获取数据
                images = batch['image']  # [B, C, H, W]
                user_indices = batch['user_idx']  # [B]
                
                # VAE编码到潜在空间
                with torch.no_grad():
                    latents = vae.encode(images).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                
                # 添加噪声
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=latents.device
                ).long()
                
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # 编码用户条件
                # 条件dropout：随机将一些条件设为无条件
                if np.random.random() < args.condition_dropout:
                    # 无条件生成：使用特殊的"无条件"token
                    user_conditions = torch.zeros_like(user_indices)
                else:
                    user_conditions = user_indices
                
                encoder_hidden_states = condition_encoder(user_conditions)
                encoder_hidden_states = encoder_hidden_states.unsqueeze(1)  # [B, 1, embed_dim]
                
                # UNet预测噪声
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=False
                )[0]
                
                # 计算损失
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
                # 反向传播
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        list(unet.parameters()) + list(condition_encoder.parameters()),
                        args.max_grad_norm
                    )
                
                optimizer.step()
                optimizer.zero_grad()
            
            # 更新进度条
            if accelerator.sync_gradients:
                global_step += 1
                epoch_loss += loss.item()
                
                # 记录日志
                if global_step % args.log_interval == 0:
                    avg_loss = epoch_loss / (step + 1)
                    
                    logs = {
                        "epoch": epoch,
                        "step": global_step,
                        "lr": lr_scheduler.get_last_lr()[0],
                        "loss": loss.item(),
                        "avg_loss": avg_loss,
                    }
                    
                    progress_bar.set_postfix(loss=f"{avg_loss:.4f}")
                    
                    if args.use_wandb and accelerator.is_main_process:
                        wandb.log(logs, step=global_step)
                
                # 生成样本图像
                if global_step % args.sample_interval == 0 and accelerator.is_main_process:
                    generate_samples(
                        unet, condition_encoder, vae, noise_scheduler,
                        data_module.all_users[:4], args.output_dir, global_step,
                        accelerator.device
                    )
        
        # 更新学习率
        lr_scheduler.step()
        
        # 验证
        if (epoch + 1) % args.val_interval == 0:
            val_loss = validate_model(
                unet, condition_encoder, vae, noise_scheduler,
                val_dataloader, accelerator
            )
            
            if args.use_wandb and accelerator.is_main_process:
                wandb.log({"val_loss": val_loss}, step=global_step)
        
        # 保存检查点
        if (epoch + 1) % args.save_interval == 0 and accelerator.is_main_process:
            save_checkpoint(
                unet, condition_encoder, optimizer, epoch, global_step,
                args.output_dir
            )
    
    # 保存最终模型
    if accelerator.is_main_process:
        save_final_model(unet, condition_encoder, args.output_dir)
        print(f"Training completed! Model saved to {args.output_dir}")

def validate_model(unet, condition_encoder, vae, noise_scheduler, val_dataloader, accelerator):
    """验证模型"""
    unet.eval()
    condition_encoder.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_dataloader:
            images = batch['image']
            user_indices = batch['user_idx']
            
            # VAE编码
            latents = vae.encode(images).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            
            # 添加噪声
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=latents.device
            ).long()
            
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # 编码条件
            encoder_hidden_states = condition_encoder(user_indices)
            encoder_hidden_states = encoder_hidden_states.unsqueeze(1)
            
            # 预测
            model_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False
            )[0]
            
            # 计算损失
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def generate_samples(unet, condition_encoder, vae, noise_scheduler, user_ids, output_dir, step, device):
    """生成样本图像"""
    unet.eval()
    condition_encoder.eval()
    
    # 创建DDIM调度器用于快速采样
    ddim_scheduler = DDIMScheduler.from_config(noise_scheduler.config)
    ddim_scheduler.set_timesteps(50)
    
    with torch.no_grad():
        # 为每个用户生成一张图像
        generated_images = []
        
        for user_id in user_ids:
            # 随机噪声
            latents = torch.randn(1, 4, 32, 32, device=device)
            
            # 用户条件
            user_idx = torch.tensor([user_id], device=device)
            encoder_hidden_states = condition_encoder(user_idx)
            encoder_hidden_states = encoder_hidden_states.unsqueeze(1)
            
            # 去噪过程
            for t in ddim_scheduler.timesteps:
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
            
            # VAE解码
            latents = latents / vae.config.scaling_factor
            image = vae.decode(latents).sample
            
            # 转换为PIL图像
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
            generated_images.append(Image.fromarray(image))
        
        # 保存图像
        sample_dir = Path(output_dir) / "samples"
        sample_dir.mkdir(exist_ok=True)
        
        # 拼接图像
        width, height = generated_images[0].size
        combined = Image.new('RGB', (width * len(generated_images), height))
        
        for i, img in enumerate(generated_images):
            combined.paste(img, (i * width, 0))
        
        combined.save(sample_dir / f"step_{step:06d}.png")

def save_checkpoint(unet, condition_encoder, optimizer, epoch, step, output_dir):
    """保存训练检查点"""
    checkpoint_dir = Path(output_dir) / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    checkpoint = {
        'unet_state_dict': unet.state_dict(),
        'condition_encoder_state_dict': condition_encoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
    }
    
    torch.save(checkpoint, checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt")

def save_final_model(unet, condition_encoder, output_dir):
    """保存最终模型"""
    final_dir = Path(output_dir) / "final_model"
    final_dir.mkdir(exist_ok=True)
    
    # 保存UNet
    unet.save_pretrained(final_dir / "unet")
    
    # 保存条件编码器
    torch.save(condition_encoder.state_dict(), final_dir / "condition_encoder.pt")

def main():
    parser = argparse.ArgumentParser(description="Train Conditional Diffusion Model for Micro-Doppler Images")
    
    # 数据参数
    parser.add_argument("--data_dir", type=str, required=True, help="数据集目录")
    parser.add_argument("--vae_path", type=str, required=True, help="预训练VAE路径")
    parser.add_argument("--resolution", type=int, default=256, help="图像分辨率")
    parser.add_argument("--val_split", type=float, default=0.2, help="验证集比例")
    
    # 模型参数
    parser.add_argument("--cross_attention_dim", type=int, default=768, help="交叉注意力维度")
    parser.add_argument("--num_train_timesteps", type=int, default=1000, help="训练时间步数")
    parser.add_argument("--condition_dropout", type=float, default=0.1, help="条件dropout概率")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--num_epochs", type=int, default=200, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="权重衰减")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪")
    
    # 系统参数
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载器工作进程数")
    parser.add_argument("--mixed_precision", type=str, default="fp16", help="混合精度")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数")
    
    # 日志和保存
    parser.add_argument("--output_dir", type=str, default="./outputs/diffusion", help="输出目录")
    parser.add_argument("--experiment_name", type=str, default="baseline", help="实验名称")
    parser.add_argument("--log_interval", type=int, default=10, help="日志记录间隔")
    parser.add_argument("--sample_interval", type=int, default=500, help="样本生成间隔")
    parser.add_argument("--val_interval", type=int, default=5, help="验证间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument("--use_wandb", action="store_true", help="使用wandb记录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    # 开始训练
    train_diffusion(args)

if __name__ == "__main__":
    main()
