#!/usr/bin/env python3
"""
微多普勒时频图 VAE 训练脚本
基于 Diffusers AutoencoderKL 实现
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL
from accelerate import Accelerator
from tqdm import tqdm
import wandb
from pathlib import Path
import argparse
import json
from PIL import Image
import numpy as np

# 导入自定义模块
import sys
sys.path.append('..')
from utils.data_loader import MicroDopplerDataset
from utils.metrics import calculate_psnr, calculate_ssim

class MicroDopplerVAELoss(nn.Module):
    """微多普勒VAE损失函数"""
    
    def __init__(self, kl_weight=1e-6, perceptual_weight=0.1, freq_weight=0.05):
        super().__init__()
        self.kl_weight = kl_weight
        self.perceptual_weight = perceptual_weight
        self.freq_weight = freq_weight
        
        # 基础损失
        self.mse_loss = nn.MSELoss()
        
        # 感知损失 (可选，需要安装 lpips)
        try:
            import lpips
            self.lpips_loss = lpips.LPIPS(net='vgg')
        except ImportError:
            print("Warning: LPIPS not available, using MSE only")
            self.lpips_loss = None
    
    def forward(self, reconstruction, target, posterior):
        """
        计算VAE损失
        Args:
            reconstruction: 重建图像 [B, C, H, W]
            target: 目标图像 [B, C, H, W]
            posterior: VAE后验分布
        """
        batch_size = target.shape[0]
        
        # 1. 重建损失
        recon_loss = self.mse_loss(reconstruction, target)
        
        # 2. KL散度损失
        kl_loss = posterior.kl().mean()
        
        # 3. 感知损失
        perceptual_loss = 0.0
        if self.lpips_loss is not None:
            # 归一化到[-1, 1]
            recon_norm = reconstruction * 2.0 - 1.0
            target_norm = target * 2.0 - 1.0
            perceptual_loss = self.lpips_loss(recon_norm, target_norm).mean()
        
        # 4. 频域损失 (保持时频特性)
        freq_loss = 0.0
        if self.freq_weight > 0:
            # FFT变换
            recon_fft = torch.fft.fft2(reconstruction)
            target_fft = torch.fft.fft2(target)
            
            # 幅度谱损失
            recon_mag = torch.abs(recon_fft)
            target_mag = torch.abs(target_fft)
            freq_loss = self.mse_loss(recon_mag, target_mag)
        
        # 总损失
        total_loss = (recon_loss + 
                     self.kl_weight * kl_loss + 
                     self.perceptual_weight * perceptual_loss + 
                     self.freq_weight * freq_loss)
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'perceptual_loss': perceptual_loss,
            'freq_loss': freq_loss
        }

def train_vae(args):
    """VAE训练主函数"""
    
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
            project="micro-doppler-vae",
            config=vars(args),
            name=f"vae-{args.experiment_name}"
        )
    
    # 创建VAE模型
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=[
            "DownEncoderBlock2D",
            "DownEncoderBlock2D", 
            "DownEncoderBlock2D",
            "DownEncoderBlock2D"
        ],
        up_block_types=[
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D"
        ],
        block_out_channels=[128, 256, 512, 512],
        latent_channels=4,
        sample_size=args.resolution,
        layers_per_block=2,
        act_fn="silu",
        norm_num_groups=32,
        scaling_factor=0.18215,
    )
    
    # 创建数据集和数据加载器
    dataset = MicroDopplerDataset(
        data_dir=args.data_dir,
        resolution=args.resolution,
        augment=args.use_augmentation
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        vae.parameters(),
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
    
    # 创建损失函数
    loss_fn = MicroDopplerVAELoss(
        kl_weight=args.kl_weight,
        perceptual_weight=args.perceptual_weight,
        freq_weight=args.freq_weight
    )
    
    # 使用accelerator准备模型和数据
    vae, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        vae, optimizer, dataloader, lr_scheduler
    )
    
    # 训练循环
    global_step = 0
    
    for epoch in range(args.num_epochs):
        vae.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(
            dataloader, 
            desc=f"Epoch {epoch+1}/{args.num_epochs}",
            disable=not accelerator.is_local_main_process
        )
        
        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(vae):
                # 获取图像数据
                images = batch['image']  # [B, C, H, W]
                
                # VAE前向传播
                posterior = vae.encode(images).latent_dist
                latents = posterior.sample()
                reconstruction = vae.decode(latents).sample
                
                # 计算损失
                loss_dict = loss_fn(reconstruction, images, posterior)
                loss = loss_dict['total_loss']
                
                # 反向传播
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(vae.parameters(), args.max_grad_norm)
                
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
                        "loss/total": loss_dict['total_loss'].item(),
                        "loss/recon": loss_dict['recon_loss'].item(),
                        "loss/kl": loss_dict['kl_loss'].item(),
                        "loss/perceptual": loss_dict['perceptual_loss'].item(),
                        "loss/freq": loss_dict['freq_loss'].item(),
                    }
                    
                    progress_bar.set_postfix(loss=f"{avg_loss:.4f}")
                    
                    if args.use_wandb and accelerator.is_main_process:
                        wandb.log(logs, step=global_step)
                
                # 保存样本图像
                if global_step % args.sample_interval == 0 and accelerator.is_main_process:
                    save_sample_images(
                        images[:4], reconstruction[:4], 
                        args.output_dir, global_step
                    )
        
        # 更新学习率
        lr_scheduler.step()
        
        # 保存检查点
        if (epoch + 1) % args.save_interval == 0 and accelerator.is_main_process:
            save_checkpoint(vae, optimizer, epoch, global_step, args.output_dir)
    
    # 保存最终模型
    if accelerator.is_main_process:
        save_final_model(vae, args.output_dir)
        print(f"Training completed! Model saved to {args.output_dir}")

def save_sample_images(original, reconstruction, output_dir, step):
    """保存样本图像"""
    import torchvision.utils as vutils
    
    # 拼接原图和重建图
    comparison = torch.cat([original, reconstruction], dim=0)
    
    # 保存图像
    sample_dir = Path(output_dir) / "samples"
    sample_dir.mkdir(exist_ok=True)
    
    vutils.save_image(
        comparison,
        sample_dir / f"step_{step:06d}.png",
        nrow=4,
        normalize=True,
        value_range=(0, 1)
    )

def save_checkpoint(model, optimizer, epoch, step, output_dir):
    """保存训练检查点"""
    checkpoint_dir = Path(output_dir) / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
    }
    
    torch.save(checkpoint, checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt")

def save_final_model(model, output_dir):
    """保存最终模型"""
    model.save_pretrained(Path(output_dir) / "final_model")

def main():
    parser = argparse.ArgumentParser(description="Train VAE for Micro-Doppler Images")
    
    # 数据参数
    parser.add_argument("--data_dir", type=str, required=True, help="数据集目录")
    parser.add_argument("--resolution", type=int, default=256, help="图像分辨率")
    parser.add_argument("--use_augmentation", action="store_true", help="使用数据增广")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--num_epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="权重衰减")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪")
    
    # 损失权重
    parser.add_argument("--kl_weight", type=float, default=1e-6, help="KL散度权重")
    parser.add_argument("--perceptual_weight", type=float, default=0.1, help="感知损失权重")
    parser.add_argument("--freq_weight", type=float, default=0.05, help="频域损失权重")
    
    # 系统参数
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载器工作进程数")
    parser.add_argument("--mixed_precision", type=str, default="fp16", help="混合精度")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数")
    
    # 日志和保存
    parser.add_argument("--output_dir", type=str, default="./outputs/vae", help="输出目录")
    parser.add_argument("--experiment_name", type=str, default="baseline", help="实验名称")
    parser.add_argument("--log_interval", type=int, default=10, help="日志记录间隔")
    parser.add_argument("--sample_interval", type=int, default=500, help="样本保存间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument("--use_wandb", action="store_true", help="使用wandb记录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    # 开始训练
    train_vae(args)

if __name__ == "__main__":
    main()
