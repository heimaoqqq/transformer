#!/usr/bin/env python3
"""
使用diffusers标准组件训练Transformer
基于成熟的、经过验证的diffusers实现
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import os
import sys

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from models.diffusers_transformer import DiffusersTransformerModel
from models.vqvae_model import MicroDopplerVQVAE
from utils.data_loader import create_micro_doppler_dataset

class DiffusersTransformerTrainer:
    """使用diffusers标准组件的Transformer训练器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🚀 初始化Diffusers Transformer训练器")
        print(f"   设备: {self.device}")
        print(f"   VQ-VAE路径: {args.vqvae_path}")
        print(f"   输出目录: {self.output_dir}")
        
        # 加载VQ-VAE模型
        self.vqvae_model = self._load_vqvae_model()
        
        # 🔒 冻结VQ-VAE模型
        print("🔒 冻结VQ-VAE模型...")
        self.vqvae_model.eval()
        for param in self.vqvae_model.parameters():
            param.requires_grad = False
        print("   ✅ VQ-VAE已冻结")
        
        # 创建Transformer模型
        self.transformer_model = self._create_transformer_model()
        
        # 创建优化器
        self.optimizer = optim.AdamW(
            self.transformer_model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95)
        )
        
        # 创建学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.num_epochs
        )
        
        print(f"✅ 训练器初始化完成")
    
    def _load_vqvae_model(self):
        """加载预训练的VQ-VAE模型"""
        vqvae_path = Path(self.args.vqvae_path)
        
        print(f"📦 加载VQ-VAE模型: {vqvae_path}")
        
        # 尝试diffusers格式
        if (vqvae_path / "config.json").exists():
            try:
                vqvae_model = MicroDopplerVQVAE.from_pretrained(vqvae_path)
                vqvae_model.to(self.device)
                print("   ✅ 成功加载diffusers格式VQ-VAE")
                return vqvae_model
            except Exception as e:
                print(f"   ❌ diffusers格式加载失败: {e}")
        
        # 尝试checkpoint格式
        checkpoint_files = list(vqvae_path.glob("*.pth"))
        if checkpoint_files:
            checkpoint_path = checkpoint_files[0]
            print(f"   📂 加载checkpoint: {checkpoint_path}")
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # 重建模型
            vqvae_model = MicroDopplerVQVAE(
                num_vq_embeddings=getattr(checkpoint.get('args', {}), 'codebook_size', 1024),
                commitment_cost=getattr(checkpoint.get('args', {}), 'commitment_cost', 0.25),
                ema_decay=getattr(checkpoint.get('args', {}), 'ema_decay', 0.99),
            )
            vqvae_model.load_state_dict(checkpoint['model_state_dict'])
            vqvae_model.to(self.device)
            print("   ✅ 成功加载checkpoint格式VQ-VAE")
            return vqvae_model
        
        raise FileNotFoundError(f"未找到VQ-VAE模型: {vqvae_path}")
    
    def _create_transformer_model(self):
        """创建diffusers标准Transformer模型"""
        print(f"🏗️ 创建Diffusers Transformer模型")
        
        model = DiffusersTransformerModel(
            vocab_size=self.args.vocab_size,
            max_seq_len=self.args.max_seq_len,
            num_users=self.args.num_users,
            num_layers=self.args.num_layers,
            num_attention_heads=self.args.num_attention_heads,
            attention_head_dim=self.args.attention_head_dim,
            dropout=self.args.dropout,
            activation_fn="gelu",
        )
        
        model.to(self.device)
        
        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   📊 模型参数:")
        print(f"      总参数: {total_params:,}")
        print(f"      可训练参数: {trainable_params:,}")
        print(f"      词汇表大小: {self.args.vocab_size}")
        print(f"      最大序列长度: {self.args.max_seq_len}")
        print(f"      用户数量: {self.args.num_users}")
        
        return model
    
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
        print(f"   批次数量: {len(dataloader)}")
        
        best_loss = float('inf')
        
        for epoch in range(self.args.num_epochs):
            print(f"\n🎯 Epoch {epoch+1}/{self.args.num_epochs}")
            
            # 确保VQ-VAE保持冻结状态
            self.vqvae_model.eval()
            
            # 训练模式
            self.transformer_model.train()
            
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
                
                # 使用VQ-VAE编码图像为token序列
                with torch.no_grad():
                    encoded = self.vqvae_model.encode(images, return_dict=True)
                    if isinstance(encoded, dict):
                        tokens = encoded['encoding_indices']
                    else:
                        tokens = encoded.encoding_indices
                    
                    # 重塑token为序列格式
                    batch_size = tokens.shape[0]
                    tokens = tokens.view(batch_size, -1)  # [B, H*W]
                
                # 准备输入和标签
                input_ids = tokens
                labels = tokens.clone()
                
                # 前向传播
                outputs = self.transformer_model(
                    input_ids=input_ids,
                    user_ids=user_ids,
                    labels=labels,
                    return_dict=True,
                )
                
                loss = outputs.loss
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    self.transformer_model.parameters(), 
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
                self._save_model(epoch, avg_loss, is_best=True)
                print(f"   ✅ 保存最佳模型 (损失: {avg_loss:.4f})")
            
            # 定期保存检查点
            if (epoch + 1) % self.args.save_every == 0:
                self._save_model(epoch, avg_loss, is_best=False)
                print(f"   💾 保存检查点")
        
        print(f"\n🎉 训练完成！最佳损失: {best_loss:.4f}")
    
    def _save_model(self, epoch, loss, is_best=False):
        """保存模型"""
        save_dict = {
            'epoch': epoch,
            'model_state_dict': self.transformer_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'args': self.args,
        }
        
        if is_best:
            save_path = self.output_dir / "best_model.pth"
        else:
            save_path = self.output_dir / f"checkpoint_epoch_{epoch+1}.pth"
        
        torch.save(save_dict, save_path)

def main():
    parser = argparse.ArgumentParser(description="训练Diffusers Transformer")
    
    # 数据参数
    parser.add_argument("--data_dir", type=str, required=True, help="数据目录")
    parser.add_argument("--vqvae_path", type=str, required=True, help="VQ-VAE模型路径")
    parser.add_argument("--output_dir", type=str, default="./diffusers_transformer_output", help="输出目录")
    
    # 模型参数
    parser.add_argument("--vocab_size", type=int, default=1024, help="词汇表大小")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="最大序列长度")
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
    
    # 创建训练器并开始训练
    trainer = DiffusersTransformerTrainer(args)
    trainer.train()

if __name__ == "__main__":
    main()
