#!/usr/bin/env python3
"""
完全基于diffusers标准组件的VQ-VAE + Transformer实现
使用diffusers.VQModel + diffusers.Transformer2DModel
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import tqdm
import sys

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

try:
    from diffusers import VQModel, Transformer2DModel
    from diffusers.configuration_utils import ConfigMixin, register_to_config
    from diffusers.models.modeling_utils import ModelMixin
    DIFFUSERS_AVAILABLE = True
    print("✅ diffusers库可用")
except ImportError:
    print("❌ diffusers库不可用，请安装最新版本: pip install diffusers")
    DIFFUSERS_AVAILABLE = False
    sys.exit(1)

from utils.data_loader import create_micro_doppler_dataset

class PureDiffusersTrainer:
    """完全基于diffusers标准组件的训练器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🚀 初始化Pure Diffusers训练器")
        print(f"   设备: {self.device}")
        print(f"   输出目录: {self.output_dir}")
        
        # 创建或加载VQ-VAE模型
        if args.vqvae_path and Path(args.vqvae_path).exists():
            print(f"📦 加载现有VQ-VAE: {args.vqvae_path}")
            self.vqvae_model = VQModel.from_pretrained(args.vqvae_path)
        else:
            print("🏗️ 创建新的diffusers VQModel")
            self.vqvae_model = VQModel(
                in_channels=3,
                out_channels=3,
                down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
                up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
                block_out_channels=[128, 256, 512],
                layers_per_block=2,
                act_fn="silu",
                latent_channels=4,
                norm_num_groups=32,
                vq_embed_dim=256,
                num_vq_embeddings=args.vocab_size,
            )
        
        self.vqvae_model.to(self.device)
        
        # 🔒 冻结VQ-VAE（如果是预训练的）
        if args.vqvae_path and Path(args.vqvae_path).exists():
            print("🔒 冻结预训练VQ-VAE")
            self.vqvae_model.eval()
            for param in self.vqvae_model.parameters():
                param.requires_grad = False
        
        # 创建Transformer模型
        print("🏗️ 创建diffusers Transformer2DModel")
        self.transformer_model = Transformer2DModel(
            num_attention_heads=args.num_attention_heads,
            attention_head_dim=args.attention_head_dim,
            in_channels=args.latent_channels,
            num_layers=args.num_layers,
            dropout=args.dropout,
            norm_num_groups=32,
            cross_attention_dim=None,  # 不使用交叉注意力
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
        self.user_embedding = nn.Embedding(args.num_users + 1, args.latent_channels)
        self.user_embedding.to(self.device)
        
        # 创建优化器
        if args.vqvae_path and Path(args.vqvae_path).exists():
            # 只优化Transformer和用户嵌入
            params = list(self.transformer_model.parameters()) + list(self.user_embedding.parameters())
        else:
            # 优化所有参数
            params = list(self.vqvae_model.parameters()) + list(self.transformer_model.parameters()) + list(self.user_embedding.parameters())
        
        self.optimizer = optim.AdamW(
            params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95)
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.num_epochs
        )
        
        print(f"✅ Pure Diffusers训练器初始化完成")
    
    def train(self):
        """训练模型"""
        print(f"🚀 开始训练...")
        
        # 创建数据集
        dataset = create_micro_doppler_dataset(
            data_dir=self.args.data_dir,
            return_user_id=True
        )
        
        # 创建数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        print(f"📊 数据集信息:")
        print(f"   样本数量: {len(dataset)}")
        print(f"   批次大小: {self.args.batch_size}")
        
        best_loss = float('inf')
        
        for epoch in range(self.args.num_epochs):
            print(f"\n🎯 Epoch {epoch+1}/{self.args.num_epochs}")
            
            # 确保VQ-VAE状态正确
            if self.args.vqvae_path and Path(self.args.vqvae_path).exists():
                self.vqvae_model.eval()  # 预训练的保持eval
            else:
                self.vqvae_model.train()  # 新建的进行训练
            
            self.transformer_model.train()
            self.user_embedding.train()
            
            total_loss = 0
            num_batches = 0
            
            pbar = tqdm(dataloader, desc=f"Training")
            
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
                
                # VQ-VAE编码
                if self.args.vqvae_path and Path(self.args.vqvae_path).exists():
                    # 预训练VQ-VAE，使用no_grad
                    with torch.no_grad():
                        vq_output = self.vqvae_model.encode(images)
                        latents = vq_output.latents
                else:
                    # 新建VQ-VAE，正常训练
                    vq_output = self.vqvae_model.encode(images)
                    latents = vq_output.latents
                
                # 添加用户条件
                batch_size, channels, height, width = latents.shape
                user_embeds = self.user_embedding(user_ids)  # [B, C]
                user_embeds = user_embeds.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
                user_embeds = user_embeds.expand(-1, -1, height, width)  # [B, C, H, W]
                
                # 组合latents和用户嵌入
                conditioned_latents = latents + user_embeds
                
                # Transformer前向传播
                transformer_output = self.transformer_model(
                    conditioned_latents,
                    return_dict=True
                )
                
                predicted_latents = transformer_output.sample
                
                # 计算重构损失
                recon_loss = nn.functional.mse_loss(predicted_latents, latents)
                
                # VQ损失（如果VQ-VAE在训练）
                vq_loss = 0
                if not (self.args.vqvae_path and Path(self.args.vqvae_path).exists()):
                    if hasattr(vq_output, 'commit_loss'):
                        vq_loss = vq_output.commit_loss.mean()
                
                # 总损失
                total_batch_loss = recon_loss + 0.25 * vq_loss
                
                # 反向传播
                self.optimizer.zero_grad()
                total_batch_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    self.optimizer.param_groups[0]['params'], 
                    max_norm=1.0
                )
                
                self.optimizer.step()
                
                # 更新统计
                total_loss += total_batch_loss.item()
                num_batches += 1
                
                # 更新进度条
                pbar.set_postfix({
                    'recon_loss': f'{recon_loss.item():.4f}',
                    'vq_loss': f'{vq_loss:.4f}' if isinstance(vq_loss, torch.Tensor) else f'{vq_loss:.4f}',
                    'total_loss': f'{total_batch_loss.item():.4f}'
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
                print(f"   ✅ 保存最佳模型 (损失: {avg_loss:.4f})")
            
            # 定期保存检查点
            if (epoch + 1) % self.args.save_every == 0:
                self._save_models(epoch, avg_loss, is_best=False)
                print(f"   💾 保存检查点")
        
        print(f"\n🎉 训练完成！最佳损失: {best_loss:.4f}")
    
    def _save_models(self, epoch, loss, is_best=False):
        """保存模型"""
        if is_best:
            # 保存VQ-VAE
            vqvae_path = self.output_dir / "vqvae_best"
            self.vqvae_model.save_pretrained(vqvae_path)
            
            # 保存Transformer
            transformer_path = self.output_dir / "transformer_best"
            self.transformer_model.save_pretrained(transformer_path)
            
            # 保存用户嵌入
            torch.save({
                'user_embedding': self.user_embedding.state_dict(),
                'epoch': epoch,
                'loss': loss,
                'args': self.args,
            }, self.output_dir / "user_embedding_best.pth")
        else:
            # 保存检查点
            checkpoint_dir = self.output_dir / f"checkpoint_epoch_{epoch+1}"
            checkpoint_dir.mkdir(exist_ok=True)
            
            self.vqvae_model.save_pretrained(checkpoint_dir / "vqvae")
            self.transformer_model.save_pretrained(checkpoint_dir / "transformer")
            
            torch.save({
                'user_embedding': self.user_embedding.state_dict(),
                'epoch': epoch,
                'loss': loss,
                'args': self.args,
            }, checkpoint_dir / "user_embedding.pth")

def main():
    parser = argparse.ArgumentParser(description="Pure Diffusers VQ-VAE + Transformer训练")
    
    # 数据参数
    parser.add_argument("--data_dir", type=str, required=True, help="数据目录")
    parser.add_argument("--vqvae_path", type=str, default=None, help="预训练VQ-VAE路径（可选）")
    parser.add_argument("--output_dir", type=str, default="./pure_diffusers_output", help="输出目录")
    
    # 模型参数
    parser.add_argument("--vocab_size", type=int, default=1024, help="VQ码本大小")
    parser.add_argument("--latent_channels", type=int, default=4, help="潜在空间通道数")
    parser.add_argument("--num_users", type=int, default=31, help="用户数量")
    parser.add_argument("--num_layers", type=int, default=8, help="Transformer层数")
    parser.add_argument("--num_attention_heads", type=int, default=8, help="注意力头数")
    parser.add_argument("--attention_head_dim", type=int, default=64, help="注意力头维度")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout率")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--num_epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载器工作进程数")
    parser.add_argument("--save_every", type=int, default=10, help="保存检查点间隔")
    
    args = parser.parse_args()
    
    print("🚀 Pure Diffusers VQ-VAE + Transformer训练")
    print("=" * 60)
    print("使用diffusers.VQModel + diffusers.Transformer2DModel")
    print("=" * 60)
    
    # 创建训练器并开始训练
    trainer = PureDiffusersTrainer(args)
    trainer.train()

if __name__ == "__main__":
    main()
