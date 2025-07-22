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
sys.path.append(str(Path(__file__).parent.parent))

from models.transformer_model import MicroDopplerTransformer
from models.vqvae_model import MicroDopplerVQVAE
from utils.data_loader import MicroDopplerDataset

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

        # 检查是否直接包含diffusers格式文件 (config.json + safetensors)
        config_file = vqvae_path / "config.json"
        safetensors_file = vqvae_path / "diffusion_pytorch_model.safetensors"

        if config_file.exists() and safetensors_file.exists():
            print(f"📂 加载VQ-VAE模型 (直接diffusers格式): {vqvae_path}")
            try:
                from models.vqvae_model import MicroDopplerVQVAE
                vqvae_model = MicroDopplerVQVAE.from_pretrained(vqvae_path)
                vqvae_model.to(self.device)
                vqvae_model.eval()
                print("✅ 成功加载直接diffusers格式模型")
                return vqvae_model
            except Exception as e:
                print(f"⚠️ 直接diffusers格式加载失败: {e}")
                print("🔄 尝试final_model子目录...")

        # 尝试final_model子目录 (diffusers格式)
        final_model_path = vqvae_path / "final_model"
        if final_model_path.exists():
            print(f"📂 加载VQ-VAE模型 (final_model子目录): {final_model_path}")
            try:
                from models.vqvae_model import MicroDopplerVQVAE
                vqvae_model = MicroDopplerVQVAE.from_pretrained(final_model_path)
                vqvae_model.to(self.device)
                vqvae_model.eval()
                print("✅ 成功加载final_model子目录格式模型")
                return vqvae_model
            except Exception as e:
                print(f"⚠️ final_model子目录格式加载失败: {e}")
                print("🔄 尝试checkpoint格式...")

        # 备选：使用checkpoint文件
        best_model_path = vqvae_path / "best_model.pth"
        if best_model_path.exists():
            model_file = best_model_path
        else:
            # 查找其他checkpoint文件
            model_files = list(vqvae_path.glob("*.pth"))
            if not model_files:
                raise FileNotFoundError(f"在 {vqvae_path} 中未找到VQ-VAE模型文件")
            model_file = model_files[0]

        print(f"📂 加载VQ-VAE模型 (checkpoint格式): {model_file}")

        # 加载checkpoint
        checkpoint = torch.load(model_file, map_location=self.device)

        # 重建VQ-VAE模型
        from models.vqvae_model import MicroDopplerVQVAE
        vqvae_model = MicroDopplerVQVAE(
            num_vq_embeddings=checkpoint['args'].codebook_size,
            commitment_cost=checkpoint['args'].commitment_cost,
            ema_decay=getattr(checkpoint['args'], 'ema_decay', 0.99),
        )

        # 加载权重
        vqvae_model.load_state_dict(checkpoint['model_state_dict'])
        vqvae_model.to(self.device)
        vqvae_model.eval()
        print("✅ 成功加载checkpoint格式模型")
        
        print(f"✅ VQ-VAE模型加载成功")
        return vqvae_model
        
    def _create_transformer_model(self):
        """创建Transformer模型"""
        model = MicroDopplerTransformer(
            vocab_size=self.args.codebook_size,
            max_seq_len=self.args.resolution * self.args.resolution // 16,  # VQ-VAE实际是4倍下采样: (128//4)^2 = 32^2 = 1024
            num_users=self.args.num_users,
            n_embd=self.args.n_embd,
            n_layer=self.args.n_layer,
            n_head=self.args.n_head,
            dropout=0.1,
            use_cross_attention=True,
        )
        model.to(self.device)

        print(f"✅ Transformer模型创建成功")
        print(f"   词汇表大小: {self.args.codebook_size}")
        print(f"   嵌入维度: {self.args.n_embd}")
        print(f"   层数: {self.args.n_layer}")
        print(f"   注意力头数: {self.args.n_head}")
        print(f"   序列长度: {model.max_seq_len}")
        print(f"   用户数量: {self.args.num_users}")

        # 测试增强功能是否工作
        self._test_enhanced_features(model)

    def _test_enhanced_features(self, model):
        """测试增强功能是否正确工作"""
        print(f"🧪 测试增强功能:")

        # 创建测试数据
        test_user_ids = torch.tensor([1, 2], device=self.device)
        test_tokens = torch.randint(0, 1024, (2, 1024), device=self.device)

        # 测试用户编码器
        with torch.no_grad():
            user_embeds = model.user_encoder(test_user_ids)
            print(f"   用户嵌入形状: {user_embeds.shape} (应该是[2, 512])")

            # 测试prepare_inputs
            input_ids, labels, encoder_hidden_states, encoder_attention_mask = model.prepare_inputs(
                test_user_ids, test_tokens
            )
            print(f"   输入序列形状: {input_ids.shape}")
            print(f"   标签形状: {labels.shape}")

            if encoder_hidden_states is not None:
                print(f"   交叉注意力状态形状: {encoder_hidden_states.shape} (应该是[2, 4, 512])")
                print(f"   注意力掩码形状: {encoder_attention_mask.shape}")
            else:
                print(f"   交叉注意力: 未使用")

        print(f"✅ 增强功能测试完成")

        return model
        
    def train(self):
        """训练Transformer"""
        print(f"\n🚀 开始Transformer训练")
        print(f"   训练轮数: {self.args.num_epochs}")
        print(f"   学习率: {self.args.learning_rate}")
        
        # 创建图像变换 - 转换为张量
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((self.args.resolution, self.args.resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到[-1, 1]
        ])

        # 创建数据加载器
        dataset = MicroDopplerDataset(
            data_dir=self.args.data_dir,
            transform=transform,  # 需要变换将PIL图像转为张量
            return_user_id=True,  # 需要用户ID进行条件生成
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

                # 用户ID范围[1,31]直接使用，嵌入层已调整为支持这个范围
                
                # 使用VQ-VAE编码图像为token序列
                with torch.no_grad():
                    encoded = self.vqvae_model.encode(images, return_dict=True)
                    tokens = encoded['encoding_indices']  # [B, H, W] - VQ-VAE输出的2D token map

                    # 检查token值范围
                    min_token = tokens.min().item()
                    max_token = tokens.max().item()
                    if min_token < 0 or max_token >= self.args.codebook_size:
                        print(f"❌ Token值超出范围: [{min_token}, {max_token}], 跳过此批次")
                        continue

                    # 展平为序列 [B, H*W] - 对于128x128图像，8倍下采样后是16x16=256
                    batch_size = tokens.shape[0]
                    tokens = tokens.view(batch_size, -1)  # [B, 256]
                
                # Transformer训练
                self.optimizer.zero_grad()
                
                # 准备输入和目标 - 确保长度匹配
                # MicroDopplerTransformer会在内部添加用户token并处理序列
                # 我们直接传递完整的token序列
                input_tokens = tokens  # 完整的token序列 [B, 1024]
                
                # 前向传播
                outputs = self.transformer_model(
                    user_ids=user_ids,
                    token_sequences=input_tokens
                )
                
                # 使用Transformer内部计算的损失
                loss = outputs.loss
                
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

    # Transformer架构参数
    parser.add_argument("--n_embd", type=int, default=512, help="Transformer嵌入维度")
    parser.add_argument("--n_layer", type=int, default=8, help="Transformer层数")
    parser.add_argument("--n_head", type=int, default=8, help="注意力头数")

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--num_epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载器工作进程数")
    parser.add_argument("--device", type=str, default="cuda", help="训练设备")

    # 保存和采样参数
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument("--sample_interval", type=int, default=10, help="样本生成间隔")
    parser.add_argument("--generation_temperature", type=float, default=1.0, help="生成温度")
    
    args = parser.parse_args()
    
    # 创建训练器并开始训练
    trainer = TransformerTrainer(args)
    trainer.train()

if __name__ == "__main__":
    main()
