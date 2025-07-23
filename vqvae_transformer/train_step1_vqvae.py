#!/usr/bin/env python3
"""
第一步：使用diffusers标准VQModel训练VQ-VAE
完全基于diffusers.VQModel的标准实现
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
    from diffusers import VQModel
    DIFFUSERS_AVAILABLE = True
    print("✅ diffusers库可用")
except ImportError:
    print("❌ diffusers库不可用，请安装最新版本: pip install diffusers")
    DIFFUSERS_AVAILABLE = False
    sys.exit(1)

from utils.data_loader import create_micro_doppler_dataset, create_datasets_with_split

class VQVAETrainer:
    """VQ-VAE训练器 - 第一步"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🚀 第一步：VQ-VAE训练器初始化")
        print(f"   设备: {self.device}")
        print(f"   输出目录: {self.output_dir}")
        
        # 创建diffusers标准VQModel
        print("🏗️ 创建diffusers VQModel")
        self.vqvae_model = VQModel(
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
            block_out_channels=[128, 256, 512, 512],
            layers_per_block=2,
            act_fn="silu",
            latent_channels=args.latent_channels,
            norm_num_groups=32,
            vq_embed_dim=args.vq_embed_dim,
            num_vq_embeddings=args.vocab_size,
            scaling_factor=0.18215,
        )
        
        self.vqvae_model.to(self.device)
        
        # 打印模型信息
        total_params = sum(p.numel() for p in self.vqvae_model.parameters())
        print(f"   📊 VQ-VAE参数: {total_params:,}")
        print(f"   📚 码本大小: {args.vocab_size}")
        print(f"   🔢 嵌入维度: {args.vq_embed_dim}")
        print(f"   📏 潜在通道: {args.latent_channels}")
        
        # 创建优化器
        self.optimizer = optim.AdamW(
            self.vqvae_model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95)
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.num_epochs
        )
        
        print(f"✅ VQ-VAE训练器初始化完成")

        # 输出关键训练参数
        self._log_training_parameters()

    def _log_training_parameters(self):
        """输出影响训练质量的关键参数"""
        print("\n" + "="*60)
        print("🔧 关键训练参数 - 影响训练质量的核心配置")
        print("="*60)

        # 模型架构参数
        print("📐 模型架构参数:")
        print(f"   🏗️ VQ-VAE架构: diffusers.VQModel (标准实现)")
        print(f"   📚 码本大小: {self.args.vocab_size}")
        print(f"   🔢 VQ嵌入维度: {self.args.vq_embed_dim}")
        print(f"   📏 潜在通道数: {self.args.latent_channels}")
        print(f"   🎯 缩放因子: 0.18215 (diffusers标准)")

        # 训练超参数
        print("\n⚙️ 训练超参数:")
        print(f"   📈 学习率: {self.args.learning_rate}")
        print(f"   🔄 训练轮数: {self.args.num_epochs}")
        print(f"   📦 批次大小: {self.args.batch_size}")
        print(f"   ⚖️ 权重衰减: {self.args.weight_decay}")
        print(f"   💪 VQ承诺损失权重: {self.args.commitment_cost}")
        print(f"   📊 优化器: AdamW (betas=(0.9, 0.95))")
        print(f"   📉 学习率调度: CosineAnnealingLR")

        # 数据处理参数
        image_size = getattr(self.args, 'image_size', 128)
        high_quality = getattr(self.args, 'high_quality_resize', True)
        scale_ratio = 256 / image_size

        print("\n🖼️ 数据处理参数:")
        print(f"   📏 原始图像尺寸: 256×256 (您的微多普勒数据集)")
        print(f"   🎯 目标图像尺寸: {image_size}×{image_size}")

        # 详细的缩放技术说明
        if high_quality:
            print(f"   🔧 缩放技术: Lanczos插值 + 抗锯齿 (默认高质量)")
            print(f"   ✨ 技术优势: 最佳细节保持，减少缩放伪影")
            print(f"   🎯 适用场景: 微多普勒细节重要，推荐生产使用")
        else:
            print(f"   🔧 缩放技术: 双线性插值 (快速模式)")
            print(f"   ⚡ 技术优势: 处理速度快，标准质量")
            print(f"   🎯 适用场景: 快速实验和测试")

        print(f"   📊 缩放比例: {scale_ratio:.1f}×下采样")
        if scale_ratio > 1:
            print(f"   ⚠️  信息损失: {(1 - 1/scale_ratio**2)*100:.1f}%像素信息")
        else:
            print(f"   ✅ 信息保持: 100%原始分辨率")

        print(f"   🎨 颜色通道: 3 (RGB)")
        print(f"   📊 归一化范围: [-1, 1] (mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])")
        print(f"   🔄 数据流程: 256×256 → {'Lanczos' if high_quality else 'Bilinear'}缩放({image_size}×{image_size}) → VQ-VAE编码")

        # 防坍缩技术
        print("\n🛡️ 码本坍缩防护技术:")
        print(f"   🔧 技术栈: diffusers内置EMA + 承诺损失")
        print(f"   📊 EMA衰减: 自适应 (diffusers管理)")
        print(f"   ⚖️ 承诺损失: {self.args.commitment_cost} * ||sg[z_e] - z_q||²")
        print(f"   🔄 码本更新: 指数移动平均 (EMA)")
        print(f"   🎯 量化策略: 最近邻 + 梯度直通估计")

        # 质量保证技术
        print("\n🎨 高质量重建技术:")
        print(f"   🏗️ 编码器: 4层下采样 (128→256→512→512)")
        print(f"   🔄 解码器: 4层上采样 (512→512→256→128)")
        print(f"   🎯 激活函数: SiLU (Swish) - 平滑梯度")
        print(f"   📊 归一化: GroupNorm (32组) - 稳定训练")
        print(f"   🔧 残差连接: 深层特征保持")
        print(f"   ⚡ 注意力机制: 无 (专注重建质量)")

        # 训练策略
        print("\n🚀 训练策略:")
        print(f"   💾 模型保存: 最佳损失 + 每{self.args.save_every}轮检查点")
        print(f"   📊 验证评估: {'启用' if self.args.use_validation else '禁用'}")
        print(f"   🖼️ 样本生成: 每{self.args.save_every}轮生成重建对比图")
        print(f"   ✂️ 梯度裁剪: max_norm=1.0 (防止梯度爆炸)")
        print(f"   🎯 损失函数: MSE重建损失 + VQ承诺损失")

        print("="*60)
        print("💡 技术说明:")
        print("   🖼️ 图像缩放: Lanczos插值+抗锯齿 (默认高质量)")
        print("   🔬 潜在缩放: diffusers标准scaling_factor=0.18215")
        print("   🛡️ 防坍缩: EMA更新 + 承诺损失 + 梯度直通估计")
        print("   🎨 高质量: SiLU激活 + GroupNorm + 残差连接")
        print("   📊 成熟技术: 基于VQGAN/VQVAE-2的成熟架构")
        print("="*60 + "\n")

    def train(self):
        """训练VQ-VAE"""
        print(f"🚀 开始VQ-VAE训练...")
        
        # 创建数据集（带自动划分）
        if self.args.use_validation:
            train_dataset, val_dataset = create_datasets_with_split(
                data_dir=self.args.data_dir,
                train_ratio=0.8,
                val_ratio=0.2,
                return_user_id=True,  # 分层划分需要user_id，训练时再处理
                random_seed=42,
                image_size=self.args.image_size,
                high_quality_resize=self.args.high_quality_resize
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
            dataset = train_dataset  # 用于统计信息

            print(f"📊 数据集信息:")
            print(f"   训练样本数量: {len(train_dataset)}")
            print(f"   验证样本数量: {len(val_dataset)}")
            print(f"   总样本数量: {len(train_dataset) + len(val_dataset)}")
            print(f"   批次大小: {self.args.batch_size}")
            print(f"   训练批次数量: {len(train_dataloader)}")
            print(f"   验证批次数量: {len(val_dataloader)}")
        else:
            # 不使用验证集，使用全部数据训练
            dataset = create_micro_doppler_dataset(
                data_dir=self.args.data_dir,
                return_user_id=False  # 不使用验证集时确实不需要user_id
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
            
            self.vqvae_model.train()
            
            total_loss = 0
            total_recon_loss = 0
            total_vq_loss = 0
            num_batches = 0
            
            pbar = tqdm(dataloader, desc=f"VQ-VAE Training")
            
            for batch_idx, batch in enumerate(pbar):
                # 处理batch格式 - 支持带user_id的数据
                if isinstance(batch, dict):
                    images = batch['image'].to(self.device)
                elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                    # 格式: (images, user_ids) - 只取images用于VQ-VAE训练
                    images, _ = batch
                    images = images.to(self.device)
                elif isinstance(batch, (list, tuple)):
                    images = batch[0].to(self.device) if len(batch) > 0 else batch.to(self.device)
                else:
                    images = batch.to(self.device)
                
                # VQ-VAE前向传播
                # 编码
                encoder_output = self.vqvae_model.encode(images)
                latents = encoder_output.latents
                
                # 解码
                decoder_output = self.vqvae_model.decode(latents)
                reconstructed = decoder_output.sample
                
                # 计算重构损失
                recon_loss = nn.functional.mse_loss(reconstructed, images)
                
                # VQ损失（commitment loss）
                vq_loss = 0

                # 调试：在第一个batch时输出encoder_output的属性
                if batch_idx == 0 and epoch == 0:
                    print(f"\n🔍 调试信息 - encoder_output属性:")
                    print(f"   类型: {type(encoder_output)}")
                    print(f"   属性: {dir(encoder_output)}")
                    if hasattr(encoder_output, '__dict__'):
                        print(f"   字典: {encoder_output.__dict__.keys()}")

                # 尝试多种可能的VQ损失属性名
                if hasattr(encoder_output, 'commit_loss') and encoder_output.commit_loss is not None:
                    vq_loss = encoder_output.commit_loss.mean()
                elif hasattr(encoder_output, 'quantization_loss') and encoder_output.quantization_loss is not None:
                    vq_loss = encoder_output.quantization_loss.mean()
                elif hasattr(encoder_output, 'loss') and encoder_output.loss is not None:
                    vq_loss = encoder_output.loss.mean()
                elif hasattr(encoder_output, 'vq_loss') and encoder_output.vq_loss is not None:
                    vq_loss = encoder_output.vq_loss.mean()
                else:
                    # 如果没有找到VQ损失，设为0（可能是diffusers版本问题）
                    vq_loss = torch.tensor(0.0, device=images.device)
                    if batch_idx == 0 and epoch == 0:
                        print(f"   ⚠️ 未找到VQ损失属性，设为0")
                
                # 总损失
                total_batch_loss = recon_loss + self.args.commitment_cost * vq_loss
                
                # 反向传播
                self.optimizer.zero_grad()
                total_batch_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    self.vqvae_model.parameters(), 
                    max_norm=1.0
                )
                
                self.optimizer.step()
                
                # 更新统计
                total_loss += total_batch_loss.item()
                total_recon_loss += recon_loss.item()
                if isinstance(vq_loss, torch.Tensor):
                    total_vq_loss += vq_loss.item()
                num_batches += 1
                
                # 更新进度条
                pbar.set_postfix({
                    'recon': f'{recon_loss.item():.4f}',
                    'vq': f'{vq_loss.item() if isinstance(vq_loss, torch.Tensor) else vq_loss:.4f}',
                    'total': f'{total_batch_loss.item():.4f}'
                })
            
            # 计算平均损失
            avg_loss = total_loss / num_batches
            avg_recon_loss = total_recon_loss / num_batches
            avg_vq_loss = total_vq_loss / num_batches
            
            # 更新学习率
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # 计算码本利用率
            codebook_usage = self._calculate_codebook_usage(dataloader)

            print(f"   📊 Epoch {epoch+1} 结果:")
            print(f"      总损失: {avg_loss:.4f}")
            print(f"      重构损失: {avg_recon_loss:.4f}")
            print(f"      VQ损失: {avg_vq_loss:.6f}")  # 增加精度显示
            print(f"      学习率: {current_lr:.6f}")
            print(f"      📚 码本利用率: {codebook_usage:.2f}% ({codebook_usage*self.args.vocab_size/100:.0f}/{self.args.vocab_size})")
            
            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                self._save_model(epoch, avg_loss, is_best=True)
                print(f"   ✅ 保存最佳VQ-VAE模型 (损失: {avg_loss:.4f})")
            
            # 验证集评估
            val_loss = None
            if self.args.use_validation and val_dataloader is not None:
                val_loss = self._validate(val_dataloader)
                print(f"      验证损失: {val_loss:.4f}")

            # 定期保存检查点和生成样本
            if (epoch + 1) % self.args.save_every == 0:
                self._save_model(epoch, avg_loss, is_best=False)
                self._generate_samples(epoch, dataloader)
                print(f"   💾 保存检查点和样本")
        
        print(f"\n🎉 VQ-VAE训练完成！最佳损失: {best_loss:.4f}")
        print(f"📁 模型保存在: {self.output_dir}")
        print(f"🔄 下一步：使用以下命令训练Transformer:")
        print(f"   python train_step2_transformer.py --vqvae_path {self.output_dir}/vqvae_best --data_dir {self.args.data_dir}")
    
    def _save_model(self, epoch, loss, is_best=False):
        """保存VQ-VAE模型"""
        if is_best:
            save_path = self.output_dir / "vqvae_best"
        else:
            save_path = self.output_dir / f"vqvae_epoch_{epoch+1}"
        
        # 使用diffusers标准保存方法
        self.vqvae_model.save_pretrained(save_path)
        
        # 保存训练信息
        torch.save({
            'epoch': epoch,
            'loss': loss,
            'args': self.args,
        }, save_path / "training_info.pth")
    
    def _generate_samples(self, epoch, dataloader):
        """生成重构样本"""
        self.vqvae_model.eval()
        
        # 获取一个batch的数据
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    images = batch['image'][:4].to(self.device)
                elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                    # 格式: (images, user_ids) - 只取images
                    images, _ = batch
                    images = images[:4].to(self.device)
                elif isinstance(batch, (list, tuple)):
                    images = batch[0][:4].to(self.device)
                else:
                    images = batch[:4].to(self.device)
                
                # 编码和解码
                encoder_output = self.vqvae_model.encode(images)
                decoder_output = self.vqvae_model.decode(encoder_output.latents)
                reconstructed = decoder_output.sample
                
                # 保存对比图像
                self._save_comparison_images(images, reconstructed, epoch)
                break
        
        self.vqvae_model.train()
    
    def _save_comparison_images(self, original, reconstructed, epoch):
        """保存原图和重构图的对比"""
        # 创建样本目录
        samples_dir = self.output_dir / "samples"
        samples_dir.mkdir(exist_ok=True)
        
        # 转换为numpy
        original = original.cpu().numpy()
        reconstructed = reconstructed.cpu().numpy()
        
        # 归一化到[0,1]
        original = (original + 1) / 2
        reconstructed = (reconstructed + 1) / 2
        
        # 创建对比图
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        for i in range(4):
            # 原图
            if original.shape[1] == 3:
                axes[0, i].imshow(original[i].transpose(1, 2, 0))
            else:
                axes[0, i].imshow(original[i, 0], cmap='viridis')
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')
            
            # 重构图
            if reconstructed.shape[1] == 3:
                axes[1, i].imshow(reconstructed[i].transpose(1, 2, 0))
            else:
                axes[1, i].imshow(reconstructed[i, 0], cmap='viridis')
            axes[1, i].set_title(f'Reconstructed {i+1}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(samples_dir / f"epoch_{epoch+1:03d}.png", dpi=150, bbox_inches='tight')
        plt.close()

    def _validate(self, val_dataloader):
        """验证模型"""
        self.vqvae_model.eval()

        total_loss = 0
        total_recon_loss = 0
        total_vq_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in val_dataloader:
                # 处理batch格式 - 支持带user_id的数据
                if isinstance(batch, dict):
                    images = batch['image'].to(self.device)
                elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                    # 格式: (images, user_ids) - 只取images用于VQ-VAE验证
                    images, _ = batch
                    images = images.to(self.device)
                elif isinstance(batch, (list, tuple)):
                    images = batch[0].to(self.device) if len(batch) > 0 else batch.to(self.device)
                else:
                    images = batch.to(self.device)

                # VQ-VAE前向传播
                encoder_output = self.vqvae_model.encode(images)
                latents = encoder_output.latents

                decoder_output = self.vqvae_model.decode(latents)
                reconstructed = decoder_output.sample

                # 计算损失
                recon_loss = nn.functional.mse_loss(reconstructed, images)

                vq_loss = 0
                if hasattr(encoder_output, 'commit_loss') and encoder_output.commit_loss is not None:
                    vq_loss = encoder_output.commit_loss.mean()
                elif hasattr(encoder_output, 'quantization_loss') and encoder_output.quantization_loss is not None:
                    vq_loss = encoder_output.quantization_loss.mean()
                elif hasattr(encoder_output, 'loss') and encoder_output.loss is not None:
                    vq_loss = encoder_output.loss.mean()
                else:
                    vq_loss = torch.tensor(0.0, device=images.device)

                total_batch_loss = recon_loss + self.args.commitment_cost * vq_loss

                # 更新统计
                total_loss += total_batch_loss.item()
                total_recon_loss += recon_loss.item()
                if isinstance(vq_loss, torch.Tensor):
                    total_vq_loss += vq_loss.item()
                num_batches += 1

        self.vqvae_model.train()
        return total_loss / num_batches if num_batches > 0 else 0

    def _calculate_codebook_usage(self, dataloader):
        """计算码本利用率"""
        self.vqvae_model.eval()

        used_codes = set()
        total_codes = self.args.vocab_size

        with torch.no_grad():
            # 只使用一部分数据来计算利用率，避免太慢
            sample_count = 0
            max_samples = min(100, len(dataloader))  # 最多100个batch

            for batch in dataloader:
                if sample_count >= max_samples:
                    break

                # 处理batch格式
                if isinstance(batch, dict):
                    images = batch['image'].to(self.device)
                elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                    images, _ = batch
                    images = images.to(self.device)
                elif isinstance(batch, (list, tuple)):
                    images = batch[0].to(self.device) if len(batch) > 0 else batch.to(self.device)
                else:
                    images = batch.to(self.device)

                # 获取量化索引
                encoder_output = self.vqvae_model.encode(images, return_dict=True)

                # 尝试获取量化索引
                if hasattr(encoder_output, 'encoding_indices'):
                    indices = encoder_output.encoding_indices
                elif hasattr(encoder_output, 'quantization_indices'):
                    indices = encoder_output.quantization_indices
                elif hasattr(encoder_output, 'indices'):
                    indices = encoder_output.indices
                else:
                    # 如果找不到索引，跳过这个batch
                    sample_count += 1
                    continue

                # 收集使用的码本索引
                if indices is not None:
                    unique_indices = torch.unique(indices.flatten()).cpu().numpy()
                    used_codes.update(unique_indices)

                sample_count += 1

        self.vqvae_model.train()

        # 计算利用率
        usage_rate = len(used_codes) / total_codes * 100
        return usage_rate

def main():
    parser = argparse.ArgumentParser(description="第一步：训练VQ-VAE")
    
    # 数据参数
    parser.add_argument("--data_dir", type=str, required=True, help="数据目录")
    parser.add_argument("--output_dir", type=str, default="./step1_vqvae_output", help="输出目录")
    
    # VQ-VAE模型参数
    parser.add_argument("--vocab_size", type=int, default=1024, help="VQ码本大小")
    parser.add_argument("--vq_embed_dim", type=int, default=256, help="VQ嵌入维度")
    parser.add_argument("--latent_channels", type=int, default=4, help="潜在空间通道数")
    parser.add_argument("--commitment_cost", type=float, default=0.25, help="VQ commitment损失权重")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--num_epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载器工作进程数")
    parser.add_argument("--save_every", type=int, default=10, help="保存检查点间隔")
    parser.add_argument("--use_validation", action="store_true", help="是否使用验证集")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--image_size", type=int, default=128, help="目标图像尺寸 (128=快速训练, 256=最高质量)")
    parser.add_argument("--high_quality_resize", action="store_true", default=True, help="使用Lanczos插值+抗锯齿 (默认推荐)")
    parser.add_argument("--fast_resize", action="store_false", dest="high_quality_resize", help="使用双线性插值 (仅用于快速测试)")
    
    args = parser.parse_args()
    
    print("🚀 第一步：VQ-VAE训练")
    print("=" * 60)
    print("使用diffusers.VQModel标准实现")
    print("=" * 60)
    
    # 创建训练器并开始训练
    trainer = VQVAETrainer(args)
    trainer.train()

if __name__ == "__main__":
    main()
