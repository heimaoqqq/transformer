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
import os
from pathlib import Path

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from ..utils.data_loader import MicroDopplerDataset
from ..utils.metrics import calculate_psnr, calculate_ssim

class MicroDopplerVAELoss(nn.Module):
    """微多普勒VAE损失函数"""
    
    def __init__(self, kl_weight=1e-4, perceptual_weight=1.0, freq_weight=0.1):
        super().__init__()
        self.kl_weight = kl_weight
        self.perceptual_weight = perceptual_weight
        self.freq_weight = freq_weight
        
        # 基础损失
        self.mse_loss = nn.MSELoss()
        
        # 感知损失 (可选，需要安装 lpips)
        self.lpips_loss = None
        if self.perceptual_weight > 0:
            try:
                import lpips
                self.lpips_loss = lpips.LPIPS(net='vgg')
                print("✅ LPIPS感知损失已加载")
            except ImportError:
                print("⚠️  LPIPS not available, disabling perceptual loss")
                self.perceptual_weight = 0.0
            except Exception as e:
                print(f"⚠️  LPIPS加载失败: {e}, disabling perceptual loss")
                self.perceptual_weight = 0.0
                self.lpips_loss = None

    def to(self, device):
        """移动损失函数到指定设备"""
        super().to(device)
        if self.lpips_loss is not None:
            self.lpips_loss = self.lpips_loss.to(device)
        return self
    
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
        if self.lpips_loss is not None and self.perceptual_weight > 0:
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

    # 内存优化设置
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # 检查GPU配置
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"🎮 检测到 {gpu_count} 个GPU")

        for i in range(gpu_count):
            gpu_props = torch.cuda.get_device_properties(i)
            gpu_memory = gpu_props.total_memory / 1024**3
            print(f"   GPU {i}: {gpu_props.name} - {gpu_memory:.1f} GB")
            torch.cuda.empty_cache()

        # 如果有多个GPU，确保使用所有GPU
        if gpu_count > 1:
            print(f"✅ 将使用所有 {gpu_count} 个GPU进行训练")
        else:
            print("⚠️  只检测到1个GPU，可能需要检查GPU配置")
    else:
        print("❌ 未检测到GPU")

    # 初始化加速器 (自动检测多GPU)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb" if args.use_wandb else None,
        project_dir=args.output_dir
    )

    # 打印实际使用的设备信息
    print(f"🚀 训练设备: {accelerator.device}")
    print(f"🔢 进程数: {accelerator.num_processes}")
    print(f"📊 是否分布式: {accelerator.distributed_type}")
    
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
    
    # 创建VAE模型 - CelebA标准配置
    # 解析架构参数
    down_blocks = args.down_block_types.split(',')
    up_blocks = args.up_block_types.split(',')
    channels = [int(c) for c in args.block_out_channels.split(',')]

    # 计算压缩比
    num_downsample = len(down_blocks)
    downsample_factor = 2 ** num_downsample
    latent_size = args.resolution // downsample_factor
    input_pixels = args.resolution * args.resolution * 3
    latent_pixels = latent_size * latent_size * args.latent_channels
    compression_ratio = input_pixels / latent_pixels

    print("🎨 使用可配置VAE架构")
    print(f"   📐 输入: {args.resolution}×{args.resolution}×3")
    print(f"   🔽 下采样层数: {num_downsample}")
    print(f"   🎯 潜在空间: {latent_size}×{latent_size}×{args.latent_channels}")
    print(f"   📊 压缩比: {compression_ratio:.1f}:1")
    print(f"   🧱 每层块数: {args.layers_per_block}")
    print(f"   📈 通道数: {channels}")

    # 确定sample_size参数
    sample_size = args.sample_size if args.sample_size is not None else args.resolution

    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=down_blocks,
        up_block_types=up_blocks,
        block_out_channels=channels,
        latent_channels=args.latent_channels,
        sample_size=sample_size,
        layers_per_block=args.layers_per_block,
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
    
    # 创建学习率调度器 - 使用更稳定的策略
    total_steps = len(dataloader) * args.num_epochs
    warmup_steps = total_steps // 10  # 10% warmup

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
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

    # 移动损失函数到正确的设备
    loss_fn = loss_fn.to(accelerator.device)
    print(f"✅ 损失函数已移动到设备: {accelerator.device}")
    
    # 训练循环
    global_step = 0
    
    for epoch in range(args.num_epochs):
        vae.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{args.num_epochs}",
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True,  # 动态调整进度条宽度
            leave=True,          # 保留进度条
            position=0           # 进度条位置
        )
        
        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(vae):
                # 获取图像数据
                images = batch['image']  # [B, C, H, W]
                
                # VAE前向传播 (处理分布式训练)
                # 在分布式训练中，需要通过.module访问原始模型
                vae_model = vae.module if hasattr(vae, 'module') else vae

                posterior = vae_model.encode(images).latent_dist
                latents = posterior.sample()
                reconstruction = vae_model.decode(latents).sample
                
                # 计算损失
                loss_dict = loss_fn(reconstruction, images, posterior)
                loss = loss_dict['total_loss']
                
                # 反向传播
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(vae.parameters(), args.max_grad_norm)
                
                optimizer.step()
                optimizer.zero_grad()

                # 定期清理GPU缓存
                if global_step % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # 更新进度条
            if accelerator.sync_gradients:
                global_step += 1
                epoch_loss += loss.item()
                
                # 记录日志
                if global_step % args.log_interval == 0:
                    avg_loss = epoch_loss / (step + 1)
                    
                    # 安全获取损失值（处理tensor和float）
                    def safe_item(value):
                        return value.item() if hasattr(value, 'item') else value

                    logs = {
                        "epoch": epoch,
                        "step": global_step,
                        "lr": lr_scheduler.get_last_lr()[0],
                        "loss/total": loss_dict['total_loss'].item(),
                        "loss/recon": loss_dict['recon_loss'].item(),
                        "loss/kl": loss_dict['kl_loss'].item(),
                        "loss/perceptual": safe_item(loss_dict['perceptual_loss']),
                        "loss/freq": safe_item(loss_dict['freq_loss']),
                    }
                    
                    # 更新进度条显示
                    progress_bar.set_postfix({
                        'loss': f"{avg_loss:.4f}",
                        'recon': f"{loss_dict['recon_loss'].item():.4f}",
                        'kl': f"{loss_dict['kl_loss'].item():.6f}",
                        'freq': f"{safe_item(loss_dict['freq_loss']):.4f}",
                        'lr': f"{lr_scheduler.get_last_lr()[0]:.2e}"
                    })
                    
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
    """保存样本图像 (只保留最近10个)"""
    import torchvision.utils as vutils

    # 拼接原图和重建图
    comparison = torch.cat([original, reconstruction], dim=0)

    # 保存图像
    sample_dir = Path(output_dir) / "samples"
    sample_dir.mkdir(exist_ok=True)

    # 新样本文件名
    new_sample_path = sample_dir / f"step_{step:06d}.png"

    # 保存新样本
    vutils.save_image(
        comparison,
        new_sample_path,
        nrow=4,
        normalize=True,
        value_range=(0, 1)
    )

    # 只保留最近10个样本文件 (节省空间)
    try:
        sample_files = sorted(sample_dir.glob("step_*.png"))
        if len(sample_files) > 10:
            # 删除最旧的文件
            for old_file in sample_files[:-10]:
                old_file.unlink()
                print(f"🗑️  删除旧样本: {old_file.name}")
    except Exception as e:
        print(f"⚠️  清理旧样本时出错: {e}")

def save_checkpoint(model, optimizer, epoch, step, output_dir):
    """保存训练检查点 (只保留最新的1个)"""
    checkpoint_dir = Path(output_dir) / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # 在分布式训练中，需要通过.module访问原始模型
    model_to_save = model.module if hasattr(model, 'module') else model

    checkpoint = {
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
    }

    # 新检查点文件名
    new_checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"

    # 删除旧的检查点文件 (保留最新1个)
    try:
        for old_checkpoint in checkpoint_dir.glob("checkpoint_epoch_*.pt"):
            if old_checkpoint != new_checkpoint_path:
                old_checkpoint.unlink()
                print(f"🗑️  删除旧检查点: {old_checkpoint.name}")
    except Exception as e:
        print(f"⚠️  删除旧检查点时出错: {e}")

    # 保存新检查点
    torch.save(checkpoint, new_checkpoint_path)
    print(f"💾 保存检查点: {new_checkpoint_path.name}")

def save_final_model(model, output_dir):
    """保存最终模型"""
    # 在分布式训练中，需要通过.module访问原始模型
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(Path(output_dir) / "final_model")

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

    # VAE架构参数
    parser.add_argument("--down_block_types", type=str, default="DownEncoderBlock2D,DownEncoderBlock2D,DownEncoderBlock2D",
                       help="下采样块类型 (逗号分隔)")
    parser.add_argument("--up_block_types", type=str, default="UpDecoderBlock2D,UpDecoderBlock2D,UpDecoderBlock2D",
                       help="上采样块类型 (逗号分隔)")
    parser.add_argument("--block_out_channels", type=str, default="64,128,256",
                       help="输出通道数 (逗号分隔)")
    parser.add_argument("--layers_per_block", type=int, default=1, help="每层块数")
    parser.add_argument("--latent_channels", type=int, default=4, help="潜在空间通道数")
    parser.add_argument("--sample_size", type=int, default=None, help="VAE sample_size参数 (影响下采样行为)")

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
