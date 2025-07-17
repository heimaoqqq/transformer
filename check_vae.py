#!/usr/bin/env python3
"""
VAE训练检查工具 - 集成版
检查训练状态、模型质量和重建效果
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from diffusers import AutoencoderKL
from utils.data_loader import MicroDopplerDataset
from torch.utils.data import DataLoader
import argparse

class VAEChecker:
    """VAE检查器"""
    
    def __init__(self, output_dir="/kaggle/working/outputs", data_dir="/kaggle/input/dataset"):
        self.output_dir = Path(output_dir)
        self.data_dir = data_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def check_training_status(self):
        """检查训练状态"""
        print("📊 VAE训练状态检查")
        print("=" * 50)
        
        if not self.output_dir.exists():
            print("❌ 输出目录不存在，可能还没有开始训练")
            print("💡 运行: python train_celeba_standard.py")
            return False
        
        # 查找训练目录
        training_dirs = [d for d in self.output_dir.iterdir() if d.is_dir()]
        
        if not training_dirs:
            print("❌ 未找到训练目录")
            return False
        
        print(f"🔍 找到 {len(training_dirs)} 个训练目录")
        
        available_models = []
        for train_dir in training_dirs:
            final_model = train_dir / "final_model"
            if final_model.exists() and (final_model / "config.json").exists():
                available_models.append(final_model)
                print(f"✅ 可用模型: {train_dir.name}")
                self._check_model_details(final_model)
        
        return len(available_models) > 0
    
    def _check_model_details(self, model_dir):
        """检查模型详细信息"""
        config_path = model_dir / "config.json"
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                print(f"   📊 模型配置:")
                print(f"     - 输入通道: {config.get('in_channels', 'N/A')}")
                print(f"     - 潜在通道: {config.get('latent_channels', 'N/A')}")
                print(f"     - 下采样层: {len(config.get('down_block_types', []))}")
                print(f"     - 通道配置: {config.get('block_out_channels', 'N/A')}")
                
            except Exception as e:
                print(f"   ❌ 配置读取失败: {e}")
    
    def test_model_loading(self, model_path=None):
        """测试模型加载"""
        print(f"\n🧪 测试模型加载")
        
        if model_path is None:
            # 自动找到第一个可用模型
            for train_dir in self.output_dir.iterdir():
                if train_dir.is_dir():
                    final_model = train_dir / "final_model"
                    if final_model.exists():
                        model_path = final_model
                        break
        
        if model_path is None:
            print("❌ 未找到可用模型")
            return None, None
        
        try:
            print(f"🔄 加载模型: {model_path}")
            vae = AutoencoderKL.from_pretrained(str(model_path))
            vae = vae.to(self.device)
            vae.eval()
            
            # 模型信息
            total_params = sum(p.numel() for p in vae.parameters())
            print(f"✅ 模型加载成功")
            print(f"   📊 参数量: {total_params:,}")
            print(f"   💾 模型大小: {total_params * 4 / 1024**2:.1f} MB")
            
            # 测试前向传播
            test_input = torch.randn(1, 3, 64, 64).to(self.device)
            with torch.no_grad():
                posterior = vae.encode(test_input).latent_dist
                latent = posterior.sample()
                output = vae.decode(latent).sample
            
            print(f"✅ 前向传播测试通过")
            print(f"   📐 压缩比: {test_input.numel() / latent.numel():.1f}:1")
            
            return vae, model_path
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return None, None
    
    def check_reconstruction_quality(self, vae=None, num_samples=8):
        """检查重建质量"""
        print(f"\n🎨 重建质量检查 ({num_samples} 张图像)")

        if vae is None:
            vae, _ = self.test_model_loading()
            if vae is None:
                return None

        try:
            # 创建测试数据
            dataset = MicroDopplerDataset(
                data_dir=self.data_dir,
                resolution=64,  # CelebA标准
                augment=False,
                split="test"
            )

            print(f"📊 测试数据: {len(dataset)} 张图像")

            # 随机选择样本
            indices = torch.randperm(len(dataset))[:num_samples]

            # 创建更大的图像网格
            fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 2, 6))

            # 如果只有一个样本，确保axes是2D数组
            if num_samples == 1:
                axes = axes.reshape(3, 1)

            mse_scores = []
            psnr_scores = []

            print(f"🔄 正在重建 {num_samples} 张图像...")

            with torch.no_grad():
                for i, idx in enumerate(indices):
                    # 获取原始图像
                    sample = dataset[idx]
                    original = sample['image'].unsqueeze(0).to(self.device)
                    user_id = sample.get('user_id', 'N/A')

                    # VAE重建
                    posterior = vae.encode(original).latent_dist
                    latent = posterior.sample()
                    reconstructed = vae.decode(latent).sample

                    # 转换为numpy
                    orig_np = original.squeeze().cpu().numpy().transpose(1, 2, 0)
                    recon_np = reconstructed.squeeze().cpu().numpy().transpose(1, 2, 0)

                    # 确保数值范围在[0,1]
                    orig_np = np.clip(orig_np, 0, 1)
                    recon_np = np.clip(recon_np, 0, 1)

                    # 计算指标
                    mse = np.mean((orig_np - recon_np) ** 2)
                    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
                    mse_scores.append(mse)
                    psnr_scores.append(psnr)

                    # 显示原始图像
                    axes[0, i].imshow(orig_np)
                    axes[0, i].set_title(f'原始 {i+1}\nUser: {user_id}', fontsize=10)
                    axes[0, i].axis('off')

                    # 显示重建图像
                    axes[1, i].imshow(recon_np)
                    axes[1, i].set_title(f'重建 {i+1}\nPSNR: {psnr:.1f}dB', fontsize=10)
                    axes[1, i].axis('off')

                    # 差异图 (热力图)
                    diff = np.abs(orig_np - recon_np)
                    # 转换为灰度用于热力图
                    diff_gray = np.mean(diff, axis=2)
                    im = axes[2, i].imshow(diff_gray, cmap='hot', vmin=0, vmax=0.3)
                    axes[2, i].set_title(f'差异 {i+1}\nMSE: {mse:.4f}', fontsize=10)
                    axes[2, i].axis('off')

                    print(f"   ✅ 样本 {i+1}: PSNR={psnr:.1f}dB, MSE={mse:.4f}")

            # 添加颜色条
            plt.colorbar(im, ax=axes[2, :], orientation='horizontal',
                        fraction=0.05, pad=0.1, label='重建误差')

            plt.suptitle(f'VAE重建质量检查 - {num_samples} 张微多普勒图像', fontsize=14, y=0.98)
            plt.tight_layout()

            # 保存高质量图像
            save_path = "/kaggle/working/vae_reconstruction_comparison.png"
            plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
            plt.show()

            print(f"✅ 重建对比图已保存: {save_path}")
            
            # 计算总体指标
            avg_mse = np.mean(mse_scores)
            psnr = 20 * np.log10(1.0 / np.sqrt(avg_mse)) if avg_mse > 0 else float('inf')
            
            # 计算相关系数
            all_orig = []
            all_recon = []
            
            with torch.no_grad():
                for idx in indices[:4]:  # 使用前4个样本计算相关性
                    sample = dataset[idx]
                    original = sample['image'].unsqueeze(0).to(self.device)
                    posterior = vae.encode(original).latent_dist
                    latent = posterior.sample()
                    reconstructed = vae.decode(latent).sample
                    
                    all_orig.append(original.cpu().numpy().flatten())
                    all_recon.append(reconstructed.cpu().numpy().flatten())
            
            correlation = np.corrcoef(np.concatenate(all_orig), np.concatenate(all_recon))[0, 1]
            
            # 显示指标
            print(f"\n📊 重建质量指标:")
            print(f"   MSE: {avg_mse:.6f}")
            print(f"   PSNR: {psnr:.2f} dB")
            print(f"   相关系数: {correlation:.4f}")
            
            # 质量评估
            print(f"\n🎯 质量评估:")
            if psnr > 25:
                print("✅ 重建质量: 优秀 (PSNR > 25dB)")
            elif psnr > 20:
                print("✅ 重建质量: 良好 (PSNR > 20dB)")
            elif psnr > 15:
                print("⚠️  重建质量: 一般 (PSNR > 15dB)")
            else:
                print("❌ 重建质量: 较差 (PSNR < 15dB)")
            
            if correlation > 0.9:
                print("✅ 相关性: 很高")
            elif correlation > 0.8:
                print("✅ 相关性: 高")
            elif correlation > 0.7:
                print("⚠️  相关性: 中等")
            else:
                print("❌ 相关性: 低")
            
            print(f"\n📁 结果保存: {save_path}")
            
            return {
                'mse': avg_mse,
                'psnr': psnr,
                'correlation': correlation
            }
            
        except Exception as e:
            print(f"❌ 重建检查失败: {e}")
            return None
    
    def full_check(self):
        """完整检查流程"""
        print("🔍 VAE完整检查")
        print("=" * 60)
        
        # 1. 检查训练状态
        if not self.check_training_status():
            print("\n💡 请先完成VAE训练:")
            print("   python train_celeba_standard.py")
            return
        
        # 2. 测试模型加载
        vae, model_path = self.test_model_loading()
        if vae is None:
            return
        
        # 3. 检查重建质量
        metrics = self.check_reconstruction_quality(vae)
        
        # 4. 总结和建议
        print(f"\n" + "=" * 60)
        print(f"📋 检查总结:")
        
        if metrics:
            if metrics['psnr'] > 20 and metrics['correlation'] > 0.8:
                print("🎉 VAE训练成功！模型质量良好")
                print("✅ 可以进行下一步扩散模型训练")
            elif metrics['psnr'] > 15:
                print("⚠️  VAE质量一般，建议优化")
                print("💡 优化建议:")
                print("   - 降低KL权重 (--kl_weight 1e-7)")
                print("   - 延长训练时间")
                print("   - 调整学习率")
            else:
                print("❌ VAE质量较差，需要重新训练")
                print("💡 重新训练建议:")
                print("   - 检查数据质量")
                print("   - 降低KL权重")
                print("   - 增加训练轮数")
        
        print(f"\n🎮 使用的设备: {self.device}")
        print(f"📁 模型路径: {model_path}")

    def analyze_latent_space(self, vae=None):
        """分析潜在空间"""
        print(f"\n🔍 潜在空间分析")

        if vae is None:
            vae, _ = self.test_model_loading()
            if vae is None:
                return

        try:
            dataset = MicroDopplerDataset(
                data_dir=self.data_dir,
                resolution=64,
                augment=False,
                split="test"
            )

            # 收集潜在向量
            latents_list = []
            num_samples = min(50, len(dataset))

            with torch.no_grad():
                for i in range(num_samples):
                    sample = dataset[i]
                    image = sample['image'].unsqueeze(0).to(self.device)
                    posterior = vae.encode(image).latent_dist
                    latent = posterior.sample()
                    latents_list.append(latent.cpu().numpy())

            latents = np.concatenate(latents_list, axis=0)

            print(f"📊 潜在空间统计 ({num_samples} 样本):")
            print(f"   形状: {latents.shape}")
            print(f"   均值: {np.mean(latents):.4f}")
            print(f"   标准差: {np.std(latents):.4f}")
            print(f"   最小值: {np.min(latents):.4f}")
            print(f"   最大值: {np.max(latents):.4f}")

            # 分析每个通道
            print(f"   各通道统计:")
            for c in range(latents.shape[1]):
                channel_data = latents[:, c, :, :]
                print(f"     通道{c}: 均值={np.mean(channel_data):.3f}, 标准差={np.std(channel_data):.3f}")

        except Exception as e:
            print(f"❌ 潜在空间分析失败: {e}")

    def generate_reconstruction_grid(self, vae=None, num_samples=8, save_individual=True):
        """生成重建图像网格，可选择保存单独图像"""
        print(f"\n🖼️  生成重建图像网格 ({num_samples} 张)")

        if vae is None:
            vae, _ = self.test_model_loading()
            if vae is None:
                return None

        try:
            dataset = MicroDopplerDataset(
                data_dir=self.data_dir,
                resolution=64,
                augment=False,
                split="test"
            )

            # 随机选择样本
            indices = torch.randperm(len(dataset))[:num_samples]

            # 创建输出目录
            output_dir = Path("/kaggle/working/reconstruction_samples")
            output_dir.mkdir(exist_ok=True)

            # 生成大网格图
            fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2.5, 5))
            if num_samples == 1:
                axes = axes.reshape(2, 1)

            all_metrics = []

            with torch.no_grad():
                for i, idx in enumerate(indices):
                    sample = dataset[idx]
                    original = sample['image'].unsqueeze(0).to(self.device)
                    user_id = sample.get('user_id', f'sample_{idx}')

                    # VAE重建
                    posterior = vae.encode(original).latent_dist
                    latent = posterior.sample()
                    reconstructed = vae.decode(latent).sample

                    # 转换为numpy
                    orig_np = original.squeeze().cpu().numpy().transpose(1, 2, 0)
                    recon_np = reconstructed.squeeze().cpu().numpy().transpose(1, 2, 0)
                    orig_np = np.clip(orig_np, 0, 1)
                    recon_np = np.clip(recon_np, 0, 1)

                    # 计算指标
                    mse = np.mean((orig_np - recon_np) ** 2)
                    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
                    all_metrics.append({'mse': mse, 'psnr': psnr, 'user_id': user_id})

                    # 网格图显示
                    axes[0, i].imshow(orig_np)
                    axes[0, i].set_title(f'原始-{i+1}\nUser: {user_id}', fontsize=9)
                    axes[0, i].axis('off')

                    axes[1, i].imshow(recon_np)
                    axes[1, i].set_title(f'重建-{i+1}\nPSNR: {psnr:.1f}dB', fontsize=9)
                    axes[1, i].axis('off')

                    # 保存单独的对比图
                    if save_individual:
                        fig_single, axes_single = plt.subplots(1, 3, figsize=(12, 4))

                        # 原始图像
                        axes_single[0].imshow(orig_np)
                        axes_single[0].set_title(f'原始图像\nUser: {user_id}', fontsize=12)
                        axes_single[0].axis('off')

                        # 重建图像
                        axes_single[1].imshow(recon_np)
                        axes_single[1].set_title(f'重建图像\nPSNR: {psnr:.1f}dB', fontsize=12)
                        axes_single[1].axis('off')

                        # 差异图
                        diff = np.abs(orig_np - recon_np)
                        diff_gray = np.mean(diff, axis=2)
                        im = axes_single[2].imshow(diff_gray, cmap='hot', vmin=0, vmax=0.3)
                        axes_single[2].set_title(f'重建误差\nMSE: {mse:.4f}', fontsize=12)
                        axes_single[2].axis('off')

                        plt.colorbar(im, ax=axes_single[2], fraction=0.046, pad=0.04)
                        plt.suptitle(f'VAE重建对比 - 样本 {i+1}', fontsize=14)
                        plt.tight_layout()

                        single_path = output_dir / f"reconstruction_sample_{i+1}_user_{user_id}.png"
                        plt.savefig(single_path, dpi=150, bbox_inches='tight', facecolor='white')
                        plt.close(fig_single)

                        print(f"   💾 保存单独图像: {single_path.name}")

            # 保存网格图
            plt.suptitle(f'VAE重建质量网格 - {num_samples} 张微多普勒图像', fontsize=14)
            plt.tight_layout()

            grid_path = "/kaggle/working/vae_reconstruction_grid.png"
            plt.savefig(grid_path, dpi=200, bbox_inches='tight', facecolor='white')
            plt.show()

            # 统计总结
            avg_psnr = np.mean([m['psnr'] for m in all_metrics])
            avg_mse = np.mean([m['mse'] for m in all_metrics])

            print(f"\n📊 生成总结:")
            print(f"   🖼️  网格图: {grid_path}")
            print(f"   📁 单独图像: {output_dir} ({num_samples} 张)")
            print(f"   📈 平均PSNR: {avg_psnr:.2f} dB")
            print(f"   📉 平均MSE: {avg_mse:.6f}")

            return all_metrics

        except Exception as e:
            print(f"❌ 图像生成失败: {e}")
            return None

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="VAE检查工具")
    parser.add_argument("--mode", choices=["status", "quick", "full", "latent", "generate"], default="full",
                       help="检查模式: status(状态), quick(快速), full(完整), latent(潜在空间), generate(生成图像)")
    parser.add_argument("--output_dir", default="/kaggle/working/outputs",
                       help="输出目录路径")
    parser.add_argument("--data_dir", default="/kaggle/input/dataset",
                       help="数据目录路径")
    parser.add_argument("--num_samples", type=int, default=8,
                       help="重建检查的样本数量")
    parser.add_argument("--save_individual", action="store_true",
                       help="是否保存单独的重建对比图")
    
    args = parser.parse_args()
    
    checker = VAEChecker(args.output_dir, args.data_dir)
    
    if args.mode == "status":
        checker.check_training_status()
    elif args.mode == "quick":
        vae, _ = checker.test_model_loading()
        if vae:
            checker.check_reconstruction_quality(vae, args.num_samples)
    elif args.mode == "latent":
        checker.analyze_latent_space()
    elif args.mode == "generate":
        vae, _ = checker.test_model_loading()
        if vae:
            checker.generate_reconstruction_grid(vae, args.num_samples, args.save_individual)
    else:  # full
        checker.full_check()

if __name__ == "__main__":
    main()
