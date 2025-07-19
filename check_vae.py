#!/usr/bin/env python3
"""
VAE训练检查工具 - 现代化版本
检查训练状态、模型质量和重建效果

更新内容:
- 支持128×128输入分辨率 (匹配train_improved_quality.py)
- 验证32×32潜在空间 (3层下采样架构)
- 提高质量评估标准 (PSNR > 28dB为优秀)
- 架构兼容性检查
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from diffusers import AutoencoderKL
from utils.data_loader import MicroDopplerDataset
import argparse

class VAEChecker:
    """VAE检查器"""

    def __init__(self, output_dir="/kaggle/working/outputs", data_dir="/kaggle/input/dataset"):
        self.output_dir = Path(output_dir)
        self.data_dir = data_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def find_model(self):
        """查找可用模型"""
        if not self.output_dir.exists():
            return None

        for train_dir in self.output_dir.iterdir():
            if train_dir.is_dir():
                final_model = train_dir / "final_model"
                if final_model.exists() and (final_model / "config.json").exists():
                    return final_model
        return None

    def load_model(self, model_path=None):
        """加载VAE模型"""
        if model_path is None:
            model_path = self.find_model()

        if model_path is None:
            print("❌ 未找到可用模型")
            return None

        try:
            print(f"🔄 加载模型: {model_path}")
            vae = AutoencoderKL.from_pretrained(str(model_path))
            vae = vae.to(self.device).eval()

            # 测试前向传播 (使用128×128匹配新训练配置)
            test_input = torch.randn(1, 3, 128, 128).to(self.device)
            with torch.no_grad():
                posterior = vae.encode(test_input).latent_dist
                latent = posterior.sample()
                _ = vae.decode(latent).sample

            total_params = sum(p.numel() for p in vae.parameters())
            print(f"✅ 模型加载成功 - 参数量: {total_params:,}")
            print(f"   📐 输入尺寸: {test_input.shape}")
            print(f"   🎯 潜在尺寸: {latent.shape}")
            print(f"   📊 压缩比: {test_input.numel() / latent.numel():.1f}:1")

            # 验证架构是否匹配新配置
            expected_latent_shape = (1, 4, 32, 32)
            if latent.shape == expected_latent_shape:
                print(f"   ✅ 架构验证: 正确的128×128→32×32配置")
            else:
                print(f"   ⚠️  架构警告: 潜在空间{latent.shape}，期望{expected_latent_shape}")
                print(f"   💡 可能是旧版本模型，建议重新训练")

            return vae

        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return None

    def check_reconstruction_quality(self, vae=None, num_samples=8):
        """检查重建质量"""
        print(f"\n🎨 重建质量检查 ({num_samples} 张图像)")

        if vae is None:
            vae = self.load_model()
            if vae is None:
                return None

        try:
            dataset = MicroDopplerDataset(
                data_dir=self.data_dir,
                resolution=128,  # 更新为128×128匹配新训练配置
                augment=False,
                split="test"
            )

            indices = torch.randperm(len(dataset))[:num_samples]
            mse_scores = []

            # 创建简单的对比图
            plt.figure(figsize=(num_samples * 3, 6))

            with torch.no_grad():
                for i, idx in enumerate(indices):
                    sample = dataset[idx]
                    original = sample['image'].unsqueeze(0).to(self.device)

                    # VAE重建
                    posterior = vae.encode(original).latent_dist
                    latent = posterior.sample()
                    reconstructed = vae.decode(latent).sample

                    # 转换为numpy并计算指标
                    orig_np = original.squeeze().cpu().numpy().transpose(1, 2, 0)
                    recon_np = reconstructed.squeeze().cpu().numpy().transpose(1, 2, 0)
                    orig_np = np.clip(orig_np, 0, 1)
                    recon_np = np.clip(recon_np, 0, 1)

                    mse = np.mean((orig_np - recon_np) ** 2)
                    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
                    mse_scores.append(mse)

                    # 显示原始图像
                    plt.subplot(2, num_samples, i + 1)
                    plt.imshow(orig_np)
                    plt.title(f'原始 {i+1}', fontsize=10)
                    plt.axis('off')

                    # 显示重建图像
                    plt.subplot(2, num_samples, i + 1 + num_samples)
                    plt.imshow(recon_np)
                    plt.title(f'重建 {i+1}\nPSNR: {psnr:.1f}dB', fontsize=10)
                    plt.axis('off')

                    print(f"   ✅ 样本 {i+1}: PSNR={psnr:.1f}dB")

            plt.suptitle('VAE重建质量检查', fontsize=14)
            plt.tight_layout()

            save_path = "/kaggle/working/vae_reconstruction.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.show()

            # 计算总体指标
            avg_mse = np.mean(mse_scores)
            avg_psnr = 20 * np.log10(1.0 / np.sqrt(avg_mse)) if avg_mse > 0 else float('inf')

            print(f"\n📊 重建质量指标:")
            print(f"   平均MSE: {avg_mse:.6f}")
            print(f"   平均PSNR: {avg_psnr:.2f} dB")

            # 质量评估 (128×128分辨率标准)
            if avg_psnr > 28:
                print("✅ 重建质量: 优秀 (现代化VAE标准)")
            elif avg_psnr > 25:
                print("✅ 重建质量: 良好")
            elif avg_psnr > 20:
                print("⚠️  重建质量: 一般")
            else:
                print("❌ 重建质量: 较差，需要重新训练")

            return {'mse': avg_mse, 'psnr': avg_psnr}

        except Exception as e:
            print(f"❌ 重建检查失败: {e}")
            return None

    def full_check(self):
        """完整检查流程"""
        print("🔍 VAE完整检查")
        print("=" * 50)

        # 检查模型
        vae = self.load_model()
        if vae is None:
            print("💡 请先完成VAE训练: python train_celeba_standard.py")
            return

        # 检查重建质量
        metrics = self.check_reconstruction_quality(vae)

        # 总结 (128×128现代化标准)
        print(f"\n📋 检查总结:")
        if metrics and metrics['psnr'] > 28:
            print("🎉 VAE训练成功！达到现代化高质量标准，可以进行下一步扩散模型训练")
        elif metrics and metrics['psnr'] > 25:
            print("✅ VAE质量良好，可以进行扩散模型训练")
        elif metrics and metrics['psnr'] > 20:
            print("⚠️  VAE质量一般，建议降低KL权重或延长训练")
        else:
            print("❌ VAE质量较差，需要重新训练或调整超参数")

    def create_simple_comparison(self, num_samples=4):
        """生成简单的左右对比图"""
        print(f"\n�️  生成对比图 ({num_samples} 张)")

        vae = self.load_model()
        if vae is None:
            return

        try:
            dataset = MicroDopplerDataset(
                data_dir=self.data_dir,
                resolution=128,  # 更新为128×128匹配新训练配置
                augment=False,
                split="test"
            )

            indices = torch.randperm(len(dataset))[:num_samples]
            output_dir = Path("/kaggle/working/comparisons")
            output_dir.mkdir(exist_ok=True)

            with torch.no_grad():
                for i, idx in enumerate(indices):
                    sample = dataset[idx]
                    original = sample['image'].unsqueeze(0).to(self.device)

                    # VAE重建
                    posterior = vae.encode(original).latent_dist
                    latent = posterior.sample()
                    reconstructed = vae.decode(latent).sample

                    # 转换为numpy
                    orig_np = original.squeeze().cpu().numpy().transpose(1, 2, 0)
                    recon_np = reconstructed.squeeze().cpu().numpy().transpose(1, 2, 0)
                    orig_np = np.clip(orig_np, 0, 1)
                    recon_np = np.clip(recon_np, 0, 1)

                    # 计算PSNR
                    mse = np.mean((orig_np - recon_np) ** 2)
                    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')

                    # 创建对比图
                    plt.figure(figsize=(10, 5))

                    plt.subplot(1, 2, 1)
                    plt.imshow(orig_np)
                    plt.title(f'原始图像 (Sample: {idx})', fontsize=14)
                    plt.axis('off')

                    plt.subplot(1, 2, 2)
                    plt.imshow(recon_np)
                    plt.title(f'重建图像 (PSNR: {psnr:.1f}dB)', fontsize=14)
                    plt.axis('off')

                    plt.suptitle(f'VAE重建对比 - 样本 {i+1}', fontsize=16)
                    plt.tight_layout()

                    save_path = output_dir / f"comparison_{i+1:02d}.png"
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close()

                    print(f"   ✅ 样本 {i+1}: PSNR={psnr:.1f}dB → {save_path.name}")

            print(f"\n📁 对比图保存在: {output_dir}")

        except Exception as e:
            print(f"❌ 对比图生成失败: {e}")



def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="VAE检查工具 - 简化版")
    parser.add_argument("--mode", choices=["check", "compare"], default="check",
                       help="模式: check(检查质量), compare(生成对比图)")
    parser.add_argument("--output_dir", default="/kaggle/working/outputs",
                       help="输出目录路径")
    parser.add_argument("--data_dir", default="/kaggle/input/dataset",
                       help="数据目录路径")
    parser.add_argument("--num_samples", type=int, default=8,
                       help="样本数量")

    args = parser.parse_args()

    checker = VAEChecker(args.output_dir, args.data_dir)

    if args.mode == "compare":
        checker.create_simple_comparison(args.num_samples)
    else:  # check
        checker.full_check()

if __name__ == "__main__":
    main()
