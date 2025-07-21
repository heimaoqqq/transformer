#!/usr/bin/env python3
"""
VQ-VAE训练脚本
第一阶段：训练VQ-VAE学习图像的离散表示
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from models.vqvae_model import MicroDopplerVQVAE
from utils.data_loader import MicroDopplerDataset
from utils.metrics import calculate_psnr, calculate_ssim

class VQVAETrainer:
    """VQ-VAE训练器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建输出目录
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置数据变换 (256x256 -> 128x128) - 使用高质量缩放
        interpolation_method = getattr(args, 'interpolation', 'lanczos')

        if interpolation_method == 'antialias':
            # 抗锯齿缩放 (推荐用于深度学习)
            resize_transform = transforms.Resize(
                (args.resolution, args.resolution),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True
            )
        else:
            # 传统插值方法
            interp_map = {
                'lanczos': transforms.InterpolationMode.LANCZOS,
                'bicubic': transforms.InterpolationMode.BICUBIC,
                'bilinear': transforms.InterpolationMode.BILINEAR,
            }
            interp_mode = interp_map.get(interpolation_method, transforms.InterpolationMode.LANCZOS)
            resize_transform = transforms.Resize((args.resolution, args.resolution), interpolation=interp_mode)

        self.transform = transforms.Compose([
            resize_transform,
            transforms.ToTensor(),  # [0, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
        ])

        print(f"🖼️ 图像缩放: 256x256 -> {args.resolution}x{args.resolution} ({interpolation_method})")
        
        # 初始化模型
        self.model = self._create_model()
        self.model.to(self.device)
        
        # 初始化优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.num_epochs,
            eta_min=args.learning_rate * 0.01,
        )
        
        # 损失函数
        self.recon_criterion = nn.MSELoss()
        
        print(f"🚀 VQ-VAE训练器初始化完成")
        print(f"   设备: {self.device}")
        print(f"   输出目录: {self.output_dir}")
        print(f"   分辨率: {args.resolution}x{args.resolution}")
    
    def _create_model(self):
        """创建VQ-VAE模型"""
        return MicroDopplerVQVAE(
            in_channels=3,
            out_channels=3,
            down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
            up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
            block_out_channels=(128, 256, 512),
            layers_per_block=2,
            act_fn="silu",
            latent_channels=256,
            sample_size=self.args.resolution // 8,  # 8倍下采样
            num_vq_embeddings=self.args.codebook_size,
            norm_num_groups=32,
            vq_embed_dim=256,
            commitment_cost=self.args.commitment_cost,
            ema_decay=self.args.ema_decay,
            restart_threshold=self.args.restart_threshold,
        )
    
    def _create_dataloader(self):
        """创建数据加载器"""
        dataset = MicroDopplerDataset(
            data_dir=self.args.data_dir,
            transform=self.transform,
            return_user_id=True,  # 返回用户ID用于后续Transformer训练
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        
        print(f"📊 数据集信息:")
        print(f"   总样本数: {len(dataset)}")
        print(f"   批次大小: {self.args.batch_size}")
        print(f"   批次数量: {len(dataloader)}")
        
        return dataloader
    
    def train_epoch(self, dataloader, epoch):
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0
        total_recon_loss = 0
        total_vq_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.args.num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                images, user_ids = batch
            else:
                images = batch
                user_ids = None
            
            images = images.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            
            outputs = self.model(images, return_dict=True)
            reconstructed = outputs.sample
            vq_loss = outputs.vq_loss
            
            # 计算重建损失
            recon_loss = self.recon_criterion(reconstructed, images)
            
            # 总损失
            total_loss_batch = recon_loss + vq_loss
            
            # 反向传播
            total_loss_batch.backward()
            
            # 梯度裁剪
            if self.args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            
            self.optimizer.step()
            
            # 统计
            total_loss += total_loss_batch.item()
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{total_loss_batch.item():.4f}',
                'Recon': f'{recon_loss.item():.4f}',
                'VQ': f'{vq_loss.item():.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}',
            })
            
            # 定期保存样本
            if batch_idx % self.args.sample_interval == 0:
                self._save_samples(images, reconstructed, epoch, batch_idx)
        
        # 更新学习率
        self.scheduler.step()
        
        # 返回平均损失
        num_batches = len(dataloader)
        return {
            'total_loss': total_loss / num_batches,
            'recon_loss': total_recon_loss / num_batches,
            'vq_loss': total_vq_loss / num_batches,
        }
    
    def _save_samples(self, original, reconstructed, epoch, batch_idx):
        """保存重建样本"""
        sample_dir = self.output_dir / "samples"
        sample_dir.mkdir(exist_ok=True)
        
        # 反归一化
        def denormalize(tensor):
            return (tensor * 0.5 + 0.5).clamp(0, 1)
        
        original = denormalize(original)
        reconstructed = denormalize(reconstructed)
        
        # 保存前4个样本
        n_samples = min(4, original.size(0))
        
        fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 3, 6))
        if n_samples == 1:
            axes = axes.reshape(2, 1)
        
        for i in range(n_samples):
            # 原图
            axes[0, i].imshow(original[i].cpu().detach().permute(1, 2, 0).numpy())
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')

            # 重建图
            axes[1, i].imshow(reconstructed[i].cpu().detach().permute(1, 2, 0).numpy())
            axes[1, i].set_title(f'Reconstructed {i+1}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(sample_dir / f"epoch_{epoch:03d}_batch_{batch_idx:04d}.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def evaluate(self, dataloader):
        """评估模型"""
        self.model.eval()
        
        total_psnr = 0
        total_ssim = 0
        num_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    images, _ = batch
                else:
                    images = batch
                
                images = images.to(self.device)
                
                outputs = self.model(images, return_dict=True)
                reconstructed = outputs.sample
                
                # 反归一化到[0,1]
                images_eval = (images * 0.5 + 0.5).clamp(0, 1)
                reconstructed_eval = (reconstructed * 0.5 + 0.5).clamp(0, 1)
                
                # 计算指标
                for i in range(images.size(0)):
                    psnr = calculate_psnr(images_eval[i], reconstructed_eval[i])
                    ssim = calculate_ssim(images_eval[i], reconstructed_eval[i])
                    
                    total_psnr += psnr
                    total_ssim += ssim
                    num_samples += 1
        
        avg_psnr = total_psnr / num_samples
        avg_ssim = total_ssim / num_samples
        
        return {'psnr': avg_psnr, 'ssim': avg_ssim}
    
    def save_model(self, epoch, is_best=False):
        """保存模型"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'args': self.args,
        }
        
        # 保存检查点
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = self.output_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"💾 保存最佳模型: {best_path}")
        
        # 保存最终模型
        final_path = self.output_dir / "final_model"
        final_path.mkdir(exist_ok=True)
        self.model.save_pretrained(final_path)
        print(f"💾 保存模型: {final_path}")
    
    def train(self):
        """主训练循环"""
        print(f"\n🎯 开始VQ-VAE训练...")
        
        # 创建数据加载器
        dataloader = self._create_dataloader()
        
        best_psnr = 0
        
        for epoch in range(self.args.num_epochs):
            # 训练
            train_metrics = self.train_epoch(dataloader, epoch)
            
            print(f"\nEpoch {epoch+1}/{self.args.num_epochs}:")
            print(f"  训练损失: {train_metrics['total_loss']:.4f}")
            print(f"  重建损失: {train_metrics['recon_loss']:.4f}")
            print(f"  VQ损失: {train_metrics['vq_loss']:.4f}")
            
            # 评估
            if (epoch + 1) % self.args.eval_interval == 0:
                eval_metrics = self.evaluate(dataloader)
                print(f"  PSNR: {eval_metrics['psnr']:.2f} dB")
                print(f"  SSIM: {eval_metrics['ssim']:.4f}")
                
                # 保存最佳模型
                is_best = eval_metrics['psnr'] > best_psnr
                if is_best:
                    best_psnr = eval_metrics['psnr']
                
                self.save_model(epoch, is_best)
            
            # 显示码本使用情况
            if (epoch + 1) % self.args.codebook_monitor_interval == 0:
                stats = self.model.get_codebook_stats()
                print(f"  📊 码本使用率: {stats['usage_rate']:.3f} ({stats['active_codes']}/{stats['total_codes']})")
                print(f"  📈 使用熵: {stats['usage_entropy']:.3f}")

                # 坍缩警告
                if stats['usage_rate'] < 0.1:
                    print(f"  ⚠️ 警告: 码本使用率过低，可能发生坍缩!")
                elif stats['usage_rate'] < 0.3:
                    print(f"  ⚠️ 注意: 码本使用率较低")
                else:
                    print(f"  ✅ 码本使用率正常")

                # 保存码本使用图
                usage_plot_path = self.output_dir / f"codebook_usage_epoch_{epoch+1:03d}.png"
                self.model.plot_codebook_usage(str(usage_plot_path))

            # 损失趋势分析
            if hasattr(self, 'loss_history'):
                self.loss_history.append(train_metrics['total_loss'])
                if len(self.loss_history) >= 3:
                    recent_trend = self.loss_history[-3:]
                    if all(recent_trend[i] < recent_trend[i+1] for i in range(len(recent_trend)-1)):
                        print(f"  ⚠️ 警告: 损失连续上升 {recent_trend}")
            else:
                self.loss_history = [train_metrics['total_loss']]
        
        print(f"\n✅ VQ-VAE训练完成!")
        print(f"   最佳PSNR: {best_psnr:.2f} dB")
        print(f"   模型保存在: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Train VQ-VAE for Micro-Doppler Images")
    
    # 数据参数
    parser.add_argument("--data_dir", type=str, required=True, help="数据集目录")
    parser.add_argument("--output_dir", type=str, default="outputs/vqvae", help="输出目录")
    parser.add_argument("--resolution", type=int, default=128, help="图像分辨率")
    parser.add_argument("--interpolation", type=str, default="lanczos",
                       choices=["lanczos", "bicubic", "bilinear", "antialias"],
                       help="图像缩放插值方法")
    
    # 模型参数
    parser.add_argument("--codebook_size", type=int, default=1024, help="码本大小")
    parser.add_argument("--commitment_cost", type=float, default=0.25, help="Commitment损失权重")
    parser.add_argument("--ema_decay", type=float, default=0.99, help="EMA衰减率")
    parser.add_argument("--restart_threshold", type=float, default=1.0, help="码本重置阈值")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--num_epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪")
    
    # 系统参数
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载器工作进程数")
    parser.add_argument("--sample_interval", type=int, default=500, help="样本保存间隔")
    parser.add_argument("--eval_interval", type=int, default=5, help="评估间隔")
    parser.add_argument("--codebook_monitor_interval", type=int, default=1, help="码本监控间隔")
    
    args = parser.parse_args()
    
    # 创建训练器并开始训练
    trainer = VQVAETrainer(args)
    trainer.train()

if __name__ == "__main__":
    main()
