#!/usr/bin/env python3
"""
Transformer训练脚本
第二阶段：训练Transformer学习从用户ID生成token序列
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from vqvae_transformer.models.transformer_model import MicroDopplerTransformer
from vqvae_transformer.models.vqvae_model import MicroDopplerVQVAE
from vqvae_transformer.utils.data_loader import MicroDopplerDataset

class TransformerTrainer:
    """Transformer训练器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🤖 Transformer训练器初始化")
        print(f"   设备: {self.device}")
        print(f"   VQ-VAE路径: {args.vqvae_path}")
        print(f"   输出目录: {self.output_dir}")
        print(f"   批次大小: {args.batch_size}")
        
        # 加载VQ-VAE模型
        self.vqvae_model = self._load_vqvae_model()
        
        # 创建Transformer模型
        self.transformer_model = self._create_transformer_model()
        
        # 创建优化器
        self.optimizer = optim.AdamW(
            self.transformer_model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # 创建学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.num_epochs
        )
        
    def _load_vqvae_model(self):
        """加载预训练的VQ-VAE模型"""
        vqvae_path = Path(self.args.vqvae_path)
        
        # 查找模型文件
        model_files = list(vqvae_path.glob("*.pth"))
        if not model_files:
            raise FileNotFoundError(f"在 {vqvae_path} 中未找到VQ-VAE模型文件")
        
        model_file = model_files[0]  # 使用第一个找到的模型文件
        print(f"📂 加载VQ-VAE模型: {model_file}")
        
        # 加载checkpoint
        checkpoint = torch.load(model_file, map_location=self.device)
        
        # 重建VQ-VAE模型
        vqvae_model = MicroDopplerVQVAE(
            num_vq_embeddings=checkpoint['args'].codebook_size,
            commitment_cost=checkpoint['args'].commitment_cost,
            ema_decay=getattr(checkpoint['args'], 'ema_decay', 0.99),
        )
        
        # 加载权重
        vqvae_model.load_state_dict(checkpoint['model_state_dict'])
        vqvae_model.to(self.device)
        vqvae_model.eval()
        
        print(f"✅ VQ-VAE模型加载成功")
        return vqvae_model
        
    def _create_transformer_model(self):
        """创建Transformer模型"""
        model = MicroDopplerTransformer(
            vocab_size=self.args.codebook_size,
            max_seq_len=self.args.resolution * self.args.resolution // 16,  # 假设16倍下采样
            d_model=512,
            nhead=8,
            num_layers=6,
            num_users=self.args.num_users,
        )
        model.to(self.device)
        
        print(f"✅ Transformer模型创建成功")
        print(f"   词汇表大小: {self.args.codebook_size}")
        print(f"   序列长度: {model.max_seq_len}")
        print(f"   用户数量: {self.args.num_users}")
        
        return model
        
    def train(self):
        """训练Transformer"""
        print(f"\n🚀 开始Transformer训练")
        print(f"   训练轮数: {self.args.num_epochs}")
        print(f"   学习率: {self.args.learning_rate}")
        
        # 创建数据加载器
        dataset = MicroDopplerDataset(
            data_dir=self.args.data_dir,
            resolution=self.args.resolution,
            num_users=self.args.num_users
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        best_loss = float('inf')
        
        for epoch in range(self.args.num_epochs):
            self.transformer_model.train()
            total_loss = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.args.num_epochs}")
            
            for batch_idx, (images, user_ids) in enumerate(pbar):
                images = images.to(self.device)
                user_ids = user_ids.to(self.device)
                
                # 使用VQ-VAE编码图像为token序列
                with torch.no_grad():
                    encoded = self.vqvae_model.encode(images)
                    tokens = encoded['encoding_indices']  # [B, H*W]
                
                # Transformer训练
                self.optimizer.zero_grad()
                
                # 准备输入和目标
                input_tokens = tokens[:, :-1]  # 除了最后一个token
                target_tokens = tokens[:, 1:]  # 除了第一个token
                
                # 前向传播
                outputs = self.transformer_model(
                    input_ids=input_tokens,
                    user_ids=user_ids
                )
                
                # 计算损失
                loss = nn.CrossEntropyLoss()(
                    outputs.logits.reshape(-1, self.args.codebook_size),
                    target_tokens.reshape(-1)
                )
                
                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.transformer_model.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                # 定期保存
                if batch_idx % 500 == 0:
                    self._save_checkpoint(epoch, batch_idx, loss.item())
            
            # 更新学习率
            self.scheduler.step()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1} 平均损失: {avg_loss:.4f}")
            
            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                self._save_model("best_model.pth", epoch, avg_loss)
        
        # 保存最终模型
        self._save_model("final_model.pth", self.args.num_epochs-1, avg_loss)
        print(f"✅ Transformer训练完成")
        
    def _save_checkpoint(self, epoch, batch_idx, loss):
        """保存训练检查点"""
        checkpoint = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'model_state_dict': self.transformer_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'args': self.args,
        }
        
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}_batch_{batch_idx}.pth"
        torch.save(checkpoint, checkpoint_path)
        
    def _save_model(self, filename, epoch, loss):
        """保存模型"""
        model_data = {
            'epoch': epoch,
            'model_state_dict': self.transformer_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'args': self.args,
        }
        
        model_path = self.output_dir / filename
        torch.save(model_data, model_path)
        print(f"💾 模型已保存: {model_path}")

def main():
    parser = argparse.ArgumentParser(description="Transformer训练脚本")
    
    # 数据参数
    parser.add_argument("--data_dir", type=str, required=True, help="数据目录")
    parser.add_argument("--vqvae_path", type=str, required=True, help="VQ-VAE模型路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    
    # 模型参数
    parser.add_argument("--resolution", type=int, default=128, help="图像分辨率")
    parser.add_argument("--codebook_size", type=int, default=1024, help="码本大小")
    parser.add_argument("--num_users", type=int, default=31, help="用户数量")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--num_epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载器工作进程数")
    parser.add_argument("--device", type=str, default="cuda", help="训练设备")
    
    args = parser.parse_args()
    
    # 创建训练器并开始训练
    trainer = TransformerTrainer(args)
    trainer.train()

if __name__ == "__main__":
    main()
