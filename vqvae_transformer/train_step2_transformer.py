#!/usr/bin/env python3
"""
第二步：使用diffusers标准Transformer2DModel训练Transformer
基于预训练的VQ-VAE模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import numpy as np

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

try:
    from diffusers import VQModel, Transformer2DModel
    DIFFUSERS_AVAILABLE = True
    print("✅ diffusers库可用")
except ImportError:
    print("❌ diffusers库不可用，请安装最新版本: pip install diffusers")
    DIFFUSERS_AVAILABLE = False
    sys.exit(1)

from utils.data_loader import create_micro_doppler_dataset, create_datasets_with_split

class TransformerTrainer:
    """Transformer训练器 - 第二步"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🚀 第二步：Transformer训练器初始化")
        print(f"   设备: {self.device}")
        print(f"   VQ-VAE路径: {args.vqvae_path}")
        print(f"   输出目录: {self.output_dir}")
        
        # 加载预训练的VQ-VAE模型
        print("📦 加载预训练VQ-VAE模型")
        self.vqvae_model = VQModel.from_pretrained(args.vqvae_path)
        self.vqvae_model.to(self.device)
        
        # 🔒 冻结VQ-VAE模型
        print("🔒 冻结VQ-VAE模型")
        self.vqvae_model.eval()
        for param in self.vqvae_model.parameters():
            param.requires_grad = False
        print("   ✅ VQ-VAE已冻结，不会更新参数")
        
        # 获取VQ-VAE的潜在空间信息
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 128, 128).to(self.device)
            dummy_output = self.vqvae_model.encode(dummy_input)
            latent_shape = dummy_output.latents.shape
            self.latent_channels = latent_shape[1]
            self.latent_height = latent_shape[2]
            self.latent_width = latent_shape[3]
        
        print(f"   📏 潜在空间形状: {self.latent_channels}x{self.latent_height}x{self.latent_width}")

        # 计算合适的norm_num_groups
        # norm_num_groups必须能整除in_channels
        possible_groups = [1, 2, 4, 8, 16, 32]
        norm_num_groups = 1
        for groups in possible_groups:
            if self.latent_channels % groups == 0:
                norm_num_groups = groups

        print(f"   🔧 使用norm_num_groups: {norm_num_groups} (适配{self.latent_channels}通道)")

        # 创建Transformer模型
        print("🏗️ 创建diffusers Transformer2DModel")
        self.transformer_model = Transformer2DModel(
            num_attention_heads=args.num_attention_heads,
            attention_head_dim=args.attention_head_dim,
            in_channels=self.latent_channels,
            num_layers=args.num_layers,
            dropout=args.dropout,
            norm_num_groups=norm_num_groups,  # 动态计算
            cross_attention_dim=args.cross_attention_dim,
            activation_fn="gelu",
            num_embeds_ada_norm=None,
            attention_bias=True,
            only_cross_attention=False,
            double_self_attention=False,
            upcast_attention=False,
            norm_elementwise_affine=True,
            norm_eps=1e-5,
            attention_type="default",
        )
        
        self.transformer_model.to(self.device)
        
        # 用户嵌入层
        print("👥 创建用户嵌入层")
        self.user_embedding = nn.Embedding(args.num_users + 1, self.latent_channels)
        self.user_embedding.to(self.device)
        
        # 输出投影层（从Transformer输出到潜在空间）
        self.output_projection = nn.Conv2d(
            self.latent_channels, 
            self.latent_channels, 
            kernel_size=1
        )
        self.output_projection.to(self.device)
        
        # 打印模型信息
        transformer_params = sum(p.numel() for p in self.transformer_model.parameters())
        user_params = sum(p.numel() for p in self.user_embedding.parameters())
        proj_params = sum(p.numel() for p in self.output_projection.parameters())
        total_trainable = transformer_params + user_params + proj_params
        
        print(f"   📊 模型参数:")
        print(f"      Transformer: {transformer_params:,}")
        print(f"      用户嵌入: {user_params:,}")
        print(f"      输出投影: {proj_params:,}")
        print(f"      总可训练参数: {total_trainable:,}")
        
        # 创建优化器（只优化Transformer相关参数）
        trainable_params = (
            list(self.transformer_model.parameters()) + 
            list(self.user_embedding.parameters()) + 
            list(self.output_projection.parameters())
        )
        
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95)
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.num_epochs
        )
        
        print(f"✅ Transformer训练器初始化完成")
    
    def train(self):
        """训练Transformer"""
        print(f"🚀 开始Transformer训练...")
        
        # 创建数据集（带自动划分）
        if self.args.use_validation:
            train_dataset, val_dataset = create_datasets_with_split(
                data_dir=self.args.data_dir,
                train_ratio=0.8,
                val_ratio=0.2,
                return_user_id=True,  # Transformer训练需要用户ID
                random_seed=42
            )

            # 创建数据加载器
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_workers,
                pin_memory=True
            )

            val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                pin_memory=True
            )

            dataloader = train_dataloader  # 主要训练用
        else:
            # 不使用验证集，使用全部数据训练
            dataset = create_micro_doppler_dataset(
                data_dir=self.args.data_dir,
                return_user_id=True  # Transformer训练需要用户ID
            )

            # 创建数据加载器
            dataloader = DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_workers,
                pin_memory=True
            )
            val_dataloader = None
        
        print(f"📊 数据集信息:")
        print(f"   样本数量: {len(dataset)}")
        print(f"   批次大小: {self.args.batch_size}")
        print(f"   批次数量: {len(dataloader)}")
        
        best_loss = float('inf')
        
        for epoch in range(self.args.num_epochs):
            print(f"\n🎯 Epoch {epoch+1}/{self.args.num_epochs}")
            
            # 确保VQ-VAE保持冻结状态
            self.vqvae_model.eval()
            
            # Transformer训练模式
            self.transformer_model.train()
            self.user_embedding.train()
            self.output_projection.train()
            
            total_loss = 0
            num_batches = 0
            
            pbar = tqdm(dataloader, desc=f"Transformer Training")
            
            for batch_idx, batch in enumerate(pbar):
                # 处理batch格式
                if isinstance(batch, dict):
                    images = batch['image'].to(self.device)
                    user_ids = batch['user_id'].to(self.device)
                elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                    images, user_ids = batch
                    images = images.to(self.device)
                    user_ids = user_ids.to(self.device)
                else:
                    continue
                
                # 使用冻结的VQ-VAE编码图像
                with torch.no_grad():
                    encoder_output = self.vqvae_model.encode(images)
                    target_latents = encoder_output.latents  # [B, C, H, W]
                
                # 添加用户条件
                batch_size = target_latents.shape[0]
                user_embeds = self.user_embedding(user_ids)  # [B, C]
                user_embeds = user_embeds.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
                user_embeds = user_embeds.expand(-1, -1, self.latent_height, self.latent_width)  # [B, C, H, W]
                
                # 组合用户条件作为输入
                conditioned_input = user_embeds
                
                # Transformer前向传播
                transformer_output = self.transformer_model(
                    conditioned_input,
                    encoder_hidden_states=None,  # 不使用交叉注意力
                    return_dict=True
                )
                
                # 输出投影
                predicted_latents = self.output_projection(transformer_output.sample)
                
                # 计算损失
                loss = nn.functional.mse_loss(predicted_latents, target_latents)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    self.optimizer.param_groups[0]['params'], 
                    max_norm=1.0
                )
                
                self.optimizer.step()
                
                # 更新统计
                total_loss += loss.item()
                num_batches += 1
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/num_batches:.4f}'
                })
            
            # 计算平均损失
            avg_loss = total_loss / num_batches
            
            # 更新学习率
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            print(f"   📊 Epoch {epoch+1} 结果:")
            print(f"      平均损失: {avg_loss:.4f}")
            print(f"      学习率: {current_lr:.6f}")
            
            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                self._save_models(epoch, avg_loss, is_best=True)
                print(f"   ✅ 保存最佳Transformer模型 (损失: {avg_loss:.4f})")
            
            # 定期保存检查点和生成样本
            if (epoch + 1) % self.args.save_every == 0:
                self._save_models(epoch, avg_loss, is_best=False)
                self._generate_samples(epoch, dataloader)
                print(f"   💾 保存检查点和样本")
        
        print(f"\n🎉 Transformer训练完成！最佳损失: {best_loss:.4f}")
        print(f"📁 模型保存在: {self.output_dir}")
    
    def _save_models(self, epoch, loss, is_best=False):
        """保存Transformer模型"""
        if is_best:
            save_dir = self.output_dir / "transformer_best"
        else:
            save_dir = self.output_dir / f"transformer_epoch_{epoch+1}"
        
        save_dir.mkdir(exist_ok=True)
        
        # 保存Transformer
        self.transformer_model.save_pretrained(save_dir / "transformer")
        
        # 保存其他组件
        torch.save({
            'user_embedding': self.user_embedding.state_dict(),
            'output_projection': self.output_projection.state_dict(),
            'epoch': epoch,
            'loss': loss,
            'args': self.args,
        }, save_dir / "additional_components.pth")
    
    def _generate_samples(self, epoch, dataloader):
        """生成样本"""
        self.transformer_model.eval()
        self.user_embedding.eval()
        self.output_projection.eval()
        
        with torch.no_grad():
            # 选择不同的用户ID
            user_ids = torch.tensor([1, 8, 16, 31], device=self.device)[:4]
            
            # 生成潜在表示
            user_embeds = self.user_embedding(user_ids)
            user_embeds = user_embeds.unsqueeze(-1).unsqueeze(-1)
            user_embeds = user_embeds.expand(-1, -1, self.latent_height, self.latent_width)
            
            # Transformer生成
            transformer_output = self.transformer_model(
                user_embeds,
                return_dict=True
            )
            
            generated_latents = self.output_projection(transformer_output.sample)
            
            # VQ-VAE解码
            decoder_output = self.vqvae_model.decode(generated_latents)
            generated_images = decoder_output.sample
            
            # 保存生成的图像
            self._save_generated_images(generated_images, user_ids, epoch)
        
        # 恢复训练模式
        self.transformer_model.train()
        self.user_embedding.train()
        self.output_projection.train()
    
    def _save_generated_images(self, images, user_ids, epoch):
        """保存生成的图像"""
        # 创建样本目录
        samples_dir = self.output_dir / "samples"
        samples_dir.mkdir(exist_ok=True)
        
        # 转换为numpy
        images = images.cpu().numpy()
        user_ids = user_ids.cpu().numpy()
        
        # 归一化到[0,1]
        images = (images + 1) / 2
        
        # 创建图像网格
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        for i in range(4):
            if images.shape[1] == 3:
                axes[i].imshow(images[i].transpose(1, 2, 0))
            else:
                axes[i].imshow(images[i, 0], cmap='viridis')
            axes[i].set_title(f'User {user_ids[i]}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(samples_dir / f"generated_epoch_{epoch+1:03d}.png", dpi=150, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="第二步：训练Transformer")
    
    # 数据参数
    parser.add_argument("--data_dir", type=str, required=True, help="数据目录")
    parser.add_argument("--vqvae_path", type=str, required=True, help="预训练VQ-VAE路径")
    parser.add_argument("--output_dir", type=str, default="./step2_transformer_output", help="输出目录")
    
    # Transformer模型参数
    parser.add_argument("--num_users", type=int, default=31, help="用户数量")
    parser.add_argument("--num_layers", type=int, default=8, help="Transformer层数")
    parser.add_argument("--num_attention_heads", type=int, default=8, help="注意力头数")
    parser.add_argument("--attention_head_dim", type=int, default=64, help="注意力头维度")
    parser.add_argument("--cross_attention_dim", type=int, default=None, help="交叉注意力维度")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout率")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--num_epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载器工作进程数")
    parser.add_argument("--save_every", type=int, default=10, help="保存检查点间隔")
    parser.add_argument("--use_validation", action="store_true", help="是否使用验证集")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="验证集比例")
    
    args = parser.parse_args()
    
    print("🚀 第二步：Transformer训练")
    print("=" * 60)
    print("使用diffusers.Transformer2DModel标准实现")
    print("基于冻结的预训练VQ-VAE")
    print("=" * 60)
    
    # 创建训练器并开始训练
    trainer = TransformerTrainer(args)
    trainer.train()

if __name__ == "__main__":
    main()
