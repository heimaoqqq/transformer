#!/usr/bin/env python3
"""
VQ-VAE质量全面评估脚本
提供多维度、准确且全面的VQ-VAE质量判断
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from PIL import Image
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from models.vqvae_model import MicroDopplerVQVAE
from utils.data_loader import MicroDopplerDataset
from utils.metrics import VQVAEMetrics, calculate_psnr, calculate_ssim

class ComprehensiveVQVAEEvaluator:
    """VQ-VAE全面质量评估器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 设置数据变换
        self.transform = transforms.Compose([
            transforms.Resize((args.resolution, args.resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # 初始化评估指标
        self.vqvae_metrics = VQVAEMetrics(device=self.device)
        
        # 创建输出目录
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🔍 VQ-VAE全面质量评估器初始化完成")
        print(f"   设备: {self.device}")
        print(f"   分辨率: {args.resolution}x{args.resolution}")
        print(f"   输出目录: {self.output_dir}")
    
    def load_model(self, model_path):
        """加载VQ-VAE模型"""
        print(f"📥 加载VQ-VAE模型: {model_path}")
        
        if Path(model_path).is_dir():
            # 从目录加载
            try:
                model = MicroDopplerVQVAE.from_pretrained(model_path)
            except:
                # 如果from_pretrained失败，尝试手动加载
                config_path = Path(model_path) / "config.json"
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    model = MicroDopplerVQVAE(**config)
                    model_file = Path(model_path) / "pytorch_model.bin"
                    if model_file.exists():
                        model.load_state_dict(torch.load(model_file, map_location=self.device))
                else:
                    raise ValueError(f"无法从目录 {model_path} 加载模型")
        else:
            # 从checkpoint加载
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 尝试从checkpoint中获取模型参数
            if 'args' in checkpoint:
                args_dict = vars(checkpoint['args']) if hasattr(checkpoint['args'], '__dict__') else checkpoint['args']
                model = MicroDopplerVQVAE(
                    in_channels=3,
                    out_channels=3,
                    sample_size=args_dict.get('resolution', 128) // 8,
                    num_vq_embeddings=args_dict.get('codebook_size', 1024),
                    commitment_cost=args_dict.get('commitment_cost', 0.25),
                    ema_decay=args_dict.get('ema_decay', 0.99),
                )
            else:
                # 使用默认参数
                model = MicroDopplerVQVAE(
                    in_channels=3,
                    out_channels=3,
                    sample_size=self.args.resolution // 8,
                    num_vq_embeddings=1024,
                )
            
            model.load_state_dict(checkpoint['model_state_dict'])
        
        model.to(self.device)
        model.eval()
        
        print(f"✅ VQ-VAE模型加载成功")
        return model
    
    def create_dataloader(self):
        """创建验证数据加载器"""
        dataset = MicroDopplerDataset(
            data_dir=self.args.data_dir,
            transform=self.transform,
            return_user_id=True,
        )
        
        # 如果指定了最大样本数，进行采样
        if self.args.max_samples and self.args.max_samples < len(dataset):
            indices = np.random.choice(len(dataset), self.args.max_samples, replace=False)
            dataset = torch.utils.data.Subset(dataset, indices)
            print(f"📊 采样验证数据集: {len(dataset)} 样本")
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        
        print(f"📊 验证数据集: {len(dataset)} 样本, {len(dataloader)} 批次")
        return dataloader
    
    def evaluate_reconstruction_quality(self, model, dataloader):
        """评估重建质量 - 核心指标"""
        print(f"\n🎯 1. 重建质量评估...")
        
        all_psnr = []
        all_ssim = []
        all_lpips = []
        all_mse = []
        all_mae = []
        
        reconstruction_errors = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="重建质量评估")):
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    images, user_ids = batch
                else:
                    images = batch
                    user_ids = None
                
                images = images.to(self.device)
                
                # 前向传播
                outputs = model(images, return_dict=True)
                reconstructed = outputs.sample
                
                # 反归一化到[0,1]进行评估
                images_eval = (images * 0.5 + 0.5).clamp(0, 1)
                reconstructed_eval = (reconstructed * 0.5 + 0.5).clamp(0, 1)
                
                # 计算各种指标
                for i in range(images.size(0)):
                    # PSNR和SSIM
                    psnr = calculate_psnr(images_eval[i], reconstructed_eval[i])
                    ssim = calculate_ssim(images_eval[i], reconstructed_eval[i])
                    
                    # MSE和MAE
                    mse = F.mse_loss(images_eval[i], reconstructed_eval[i]).item()
                    mae = F.l1_loss(images_eval[i], reconstructed_eval[i]).item()
                    
                    all_psnr.append(psnr)
                    all_ssim.append(ssim)
                    all_mse.append(mse)
                    all_mae.append(mae)
                    
                    # 记录重建误差分布
                    error = torch.abs(images_eval[i] - reconstructed_eval[i]).mean().item()
                    reconstruction_errors.append(error)
                    
                    # LPIPS (如果可用)
                    if self.vqvae_metrics.lpips_model is not None:
                        try:
                            lpips_score = self.vqvae_metrics.lpips_model(
                                images_eval[i:i+1] * 2 - 1,  # 转换到[-1,1]
                                reconstructed_eval[i:i+1] * 2 - 1
                            ).item()
                            all_lpips.append(lpips_score)
                        except:
                            pass
                
                # 保存样本
                if batch_idx < 5:  # 保存前5个batch的样本
                    self._save_reconstruction_samples(images_eval, reconstructed_eval, batch_idx)
        
        # 计算统计信息
        results = {
            'psnr': {
                'mean': np.mean(all_psnr),
                'std': np.std(all_psnr),
                'min': np.min(all_psnr),
                'max': np.max(all_psnr),
                'median': np.median(all_psnr),
                'q25': np.percentile(all_psnr, 25),
                'q75': np.percentile(all_psnr, 75),
            },
            'ssim': {
                'mean': np.mean(all_ssim),
                'std': np.std(all_ssim),
                'min': np.min(all_ssim),
                'max': np.max(all_ssim),
                'median': np.median(all_ssim),
                'q25': np.percentile(all_ssim, 25),
                'q75': np.percentile(all_ssim, 75),
            },
            'mse': {
                'mean': np.mean(all_mse),
                'std': np.std(all_mse),
                'min': np.min(all_mse),
                'max': np.max(all_mse),
            },
            'mae': {
                'mean': np.mean(all_mae),
                'std': np.std(all_mae),
                'min': np.min(all_mae),
                'max': np.max(all_mae),
            },
            'reconstruction_error_distribution': reconstruction_errors,
        }
        
        if all_lpips:
            results['lpips'] = {
                'mean': np.mean(all_lpips),
                'std': np.std(all_lpips),
                'min': np.min(all_lpips),
                'max': np.max(all_lpips),
            }
        
        # 绘制分布图
        self._plot_reconstruction_metrics(all_psnr, all_ssim, all_mse, reconstruction_errors)

        return results

    def evaluate_codebook_quality(self, model, dataloader):
        """评估码本质量 - 关键指标"""
        print(f"\n🎯 2. 码本质量评估...")

        # 收集所有编码索引和特征
        all_indices = []
        all_features = []
        codebook_embeddings = model.quantize.embedding.weight.data.cpu().numpy()

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="码本质量评估"):
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    images, user_ids = batch
                else:
                    images = batch

                images = images.to(self.device)

                # 编码
                encode_result = model.encode(images, return_dict=True)
                indices = encode_result['encoding_indices']  # [B, H, W]
                latents = encode_result['latents']  # [B, C, H, W]

                # 收集索引和特征
                all_indices.extend(indices.cpu().flatten().numpy())
                all_features.extend(latents.cpu().view(latents.size(0), -1).numpy())

        all_indices = np.array(all_indices)
        all_features = np.array(all_features)

        # 1. 码本使用统计
        unique_indices, counts = np.unique(all_indices, return_counts=True)
        usage_rate = len(unique_indices) / model.quantize.n_embed

        # 2. 使用分布熵
        usage_probs = counts / counts.sum()
        usage_entropy = entropy(usage_probs)
        max_entropy = np.log(len(unique_indices))
        normalized_entropy = usage_entropy / max_entropy if max_entropy > 0 else 0

        # 3. 码本向量聚类质量
        if len(unique_indices) > 1:
            used_embeddings = codebook_embeddings[unique_indices]
            if len(used_embeddings) >= 2:
                try:
                    # 使用K-means评估聚类质量
                    n_clusters = min(10, len(used_embeddings))
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(used_embeddings)
                    silhouette_avg = silhouette_score(used_embeddings, cluster_labels)
                except:
                    silhouette_avg = 0.0
            else:
                silhouette_avg = 0.0
        else:
            silhouette_avg = 0.0

        # 4. 码本向量间距离分析
        if len(unique_indices) > 1:
            used_embeddings = codebook_embeddings[unique_indices]
            # 计算所有向量对之间的距离
            distances = []
            for i in range(len(used_embeddings)):
                for j in range(i+1, len(used_embeddings)):
                    dist = np.linalg.norm(used_embeddings[i] - used_embeddings[j])
                    distances.append(dist)

            min_distance = np.min(distances) if distances else 0
            mean_distance = np.mean(distances) if distances else 0
            std_distance = np.std(distances) if distances else 0
        else:
            min_distance = mean_distance = std_distance = 0

        # 5. 重建一致性检查
        reconstruction_consistency = self._check_reconstruction_consistency(model, dataloader)

        results = {
            'usage_statistics': {
                'total_codes': model.quantize.n_embed,
                'used_codes': len(unique_indices),
                'usage_rate': usage_rate,
                'unused_codes': model.quantize.n_embed - len(unique_indices),
            },
            'usage_distribution': {
                'entropy': usage_entropy,
                'normalized_entropy': normalized_entropy,
                'max_entropy': max_entropy,
                'gini_coefficient': self._calculate_gini_coefficient(counts),
            },
            'clustering_quality': {
                'silhouette_score': silhouette_avg,
            },
            'distance_analysis': {
                'min_distance': min_distance,
                'mean_distance': mean_distance,
                'std_distance': std_distance,
            },
            'reconstruction_consistency': reconstruction_consistency,
        }

        # 绘制码本分析图
        self._plot_codebook_analysis(unique_indices, counts, codebook_embeddings, used_embeddings if len(unique_indices) > 1 else None)

        return results

    def evaluate_latent_space_quality(self, model, dataloader):
        """评估潜在空间质量"""
        print(f"\n🎯 3. 潜在空间质量评估...")

        all_latents = []
        all_user_ids = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="潜在空间评估"):
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    images, user_ids = batch
                else:
                    images = batch
                    user_ids = None

                images = images.to(self.device)

                # 编码到潜在空间
                encode_result = model.encode(images, return_dict=True)
                latents = encode_result['latents']  # [B, C, H, W]

                # 展平潜在表示
                latents_flat = latents.view(latents.size(0), -1)
                all_latents.append(latents_flat.cpu())

                if user_ids is not None:
                    all_user_ids.extend(user_ids.cpu().numpy())

        all_latents = torch.cat(all_latents, dim=0).numpy()

        # 1. 潜在空间的统计特性
        latent_mean = np.mean(all_latents, axis=0)
        latent_std = np.std(all_latents, axis=0)
        latent_var = np.var(all_latents, axis=0)

        # 2. 潜在空间的分布特性
        latent_range = np.max(all_latents) - np.min(all_latents)
        latent_sparsity = np.mean(np.abs(all_latents) < 0.1)  # 接近0的比例

        # 3. 如果有用户ID，评估用户分离度
        user_separation = None
        if all_user_ids:
            user_separation = self._evaluate_user_separation(all_latents, all_user_ids)

        results = {
            'statistical_properties': {
                'mean_magnitude': np.mean(np.abs(latent_mean)),
                'std_magnitude': np.mean(latent_std),
                'variance_magnitude': np.mean(latent_var),
                'range': latent_range,
                'sparsity': latent_sparsity,
            },
            'user_separation': user_separation,
        }

        # 绘制潜在空间分析图
        self._plot_latent_space_analysis(all_latents, all_user_ids)

        return results

    def _check_reconstruction_consistency(self, model, dataloader):
        """检查重建一致性"""
        consistency_scores = []

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    images, _ = batch
                else:
                    images = batch

                images = images.to(self.device)

                # 第一次重建
                outputs1 = model(images, return_dict=True)
                reconstructed1 = outputs1.sample

                # 第二次重建（应该完全一致）
                outputs2 = model(images, return_dict=True)
                reconstructed2 = outputs2.sample

                # 计算一致性
                consistency = F.mse_loss(reconstructed1, reconstructed2).item()
                consistency_scores.append(consistency)

                # 只检查几个batch
                if len(consistency_scores) >= 5:
                    break

        return {
            'mean_consistency_error': np.mean(consistency_scores),
            'max_consistency_error': np.max(consistency_scores),
            'is_deterministic': np.max(consistency_scores) < 1e-6,
        }

    def _calculate_gini_coefficient(self, counts):
        """计算基尼系数，衡量使用分布的不均匀程度"""
        if len(counts) <= 1:
            return 0.0

        # 排序
        sorted_counts = np.sort(counts)
        n = len(sorted_counts)

        # 计算基尼系数
        cumsum = np.cumsum(sorted_counts)
        gini = (2 * np.sum((np.arange(1, n + 1) * sorted_counts))) / (n * cumsum[-1]) - (n + 1) / n

        return gini

    def _evaluate_user_separation(self, latents, user_ids):
        """评估用户在潜在空间中的分离度"""
        unique_users = np.unique(user_ids)
        if len(unique_users) < 2:
            return None

        # 计算类内和类间距离
        intra_class_distances = []
        inter_class_distances = []

        for user in unique_users:
            user_latents = latents[np.array(user_ids) == user]

            if len(user_latents) > 1:
                # 类内距离
                for i in range(len(user_latents)):
                    for j in range(i+1, len(user_latents)):
                        dist = np.linalg.norm(user_latents[i] - user_latents[j])
                        intra_class_distances.append(dist)

            # 类间距离
            other_users = unique_users[unique_users != user]
            for other_user in other_users:
                other_latents = latents[np.array(user_ids) == other_user]
                for user_latent in user_latents:
                    for other_latent in other_latents:
                        dist = np.linalg.norm(user_latent - other_latent)
                        inter_class_distances.append(dist)

        if not intra_class_distances or not inter_class_distances:
            return None

        mean_intra = np.mean(intra_class_distances)
        mean_inter = np.mean(inter_class_distances)
        separation_ratio = mean_inter / mean_intra if mean_intra > 0 else float('inf')

        return {
            'mean_intra_class_distance': mean_intra,
            'mean_inter_class_distance': mean_inter,
            'separation_ratio': separation_ratio,
            'num_users': len(unique_users),
        }

    def _save_reconstruction_samples(self, original, reconstructed, batch_idx):
        """保存重建样本"""
        sample_dir = self.output_dir / "reconstruction_samples"
        sample_dir.mkdir(exist_ok=True)

        # 保存前4个样本
        n_samples = min(4, original.size(0))

        fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 3, 6))
        if n_samples == 1:
            axes = axes.reshape(2, 1)

        for i in range(n_samples):
            # 原图
            axes[0, i].imshow(original[i].cpu().permute(1, 2, 0).numpy())
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')

            # 重建图
            axes[1, i].imshow(reconstructed[i].cpu().permute(1, 2, 0).numpy())
            axes[1, i].set_title(f'Reconstructed {i+1}')
            axes[1, i].axis('off')

        plt.tight_layout()
        plt.savefig(sample_dir / f"batch_{batch_idx:04d}.png", dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_reconstruction_metrics(self, psnr_values, ssim_values, mse_values, error_values):
        """绘制重建指标分布图"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # PSNR分布
        axes[0, 0].hist(psnr_values, bins=50, alpha=0.7, color='blue')
        axes[0, 0].set_xlabel('PSNR (dB)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title(f'PSNR Distribution\nMean: {np.mean(psnr_values):.2f} ± {np.std(psnr_values):.2f}')
        axes[0, 0].axvline(np.mean(psnr_values), color='red', linestyle='--', label='Mean')
        axes[0, 0].legend()

        # SSIM分布
        axes[0, 1].hist(ssim_values, bins=50, alpha=0.7, color='green')
        axes[0, 1].set_xlabel('SSIM')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title(f'SSIM Distribution\nMean: {np.mean(ssim_values):.4f} ± {np.std(ssim_values):.4f}')
        axes[0, 1].axvline(np.mean(ssim_values), color='red', linestyle='--', label='Mean')
        axes[0, 1].legend()

        # MSE分布
        axes[1, 0].hist(mse_values, bins=50, alpha=0.7, color='orange')
        axes[1, 0].set_xlabel('MSE')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title(f'MSE Distribution\nMean: {np.mean(mse_values):.6f}')
        axes[1, 0].set_yscale('log')

        # 重建误差分布
        axes[1, 1].hist(error_values, bins=50, alpha=0.7, color='purple')
        axes[1, 1].set_xlabel('Reconstruction Error')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title(f'Reconstruction Error Distribution\nMean: {np.mean(error_values):.6f}')

        plt.tight_layout()
        plt.savefig(self.output_dir / "reconstruction_metrics.png", dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_codebook_analysis(self, unique_indices, counts, all_embeddings, used_embeddings):
        """绘制码本分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. 码本使用分布
        axes[0, 0].bar(range(len(counts)), sorted(counts, reverse=True))
        axes[0, 0].set_xlabel('Code Index (sorted by usage)')
        axes[0, 0].set_ylabel('Usage Count')
        axes[0, 0].set_title(f'Codebook Usage Distribution\n{len(unique_indices)}/{len(all_embeddings)} codes used')
        axes[0, 0].set_yscale('log')

        # 2. 使用频率直方图
        axes[0, 1].hist(counts, bins=min(50, len(counts)), alpha=0.7)
        axes[0, 1].set_xlabel('Usage Count')
        axes[0, 1].set_ylabel('Number of Codes')
        axes[0, 1].set_title('Usage Frequency Histogram')

        # 3. 码本向量的2D投影（如果有使用的向量）
        if used_embeddings is not None and len(used_embeddings) > 1:
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                embeddings_2d = pca.fit_transform(used_embeddings)

                scatter = axes[1, 0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                                           c=counts, cmap='viridis', alpha=0.7)
                axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
                axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
                axes[1, 0].set_title('Codebook Embeddings (PCA)')
                plt.colorbar(scatter, ax=axes[1, 0], label='Usage Count')
            except:
                axes[1, 0].text(0.5, 0.5, 'PCA failed',
                              ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Codebook Embeddings (PCA)')
        else:
            axes[1, 0].text(0.5, 0.5, 'Insufficient data for PCA',
                          ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Codebook Embeddings (PCA)')

        # 4. 累积使用率
        sorted_counts = sorted(counts, reverse=True)
        cumulative_usage = np.cumsum(sorted_counts) / np.sum(sorted_counts)
        axes[1, 1].plot(range(1, len(cumulative_usage) + 1), cumulative_usage)
        axes[1, 1].set_xlabel('Number of Most Used Codes')
        axes[1, 1].set_ylabel('Cumulative Usage Ratio')
        axes[1, 1].set_title('Cumulative Usage Distribution')
        axes[1, 1].grid(True, alpha=0.3)

        # 添加80%使用率线
        if len(cumulative_usage) > 0:
            idx_80 = np.where(cumulative_usage >= 0.8)[0]
            if len(idx_80) > 0:
                axes[1, 1].axvline(idx_80[0] + 1, color='red', linestyle='--',
                                 label=f'80% usage: {idx_80[0] + 1} codes')
                axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / "codebook_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_latent_space_analysis(self, latents, user_ids):
        """绘制潜在空间分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. 潜在向量的统计分布
        latent_means = np.mean(latents, axis=0)
        axes[0, 0].hist(latent_means, bins=50, alpha=0.7)
        axes[0, 0].set_xlabel('Mean Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Latent Dimension Means')

        # 2. 潜在向量的方差分布
        latent_vars = np.var(latents, axis=0)
        axes[0, 1].hist(latent_vars, bins=50, alpha=0.7)
        axes[0, 1].set_xlabel('Variance')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Latent Dimension Variances')
        axes[0, 1].set_yscale('log')

        # 3. 潜在空间的2D投影
        if latents.shape[0] > 1:
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                latents_2d = pca.fit_transform(latents)

                if user_ids:
                    unique_users = np.unique(user_ids)
                    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_users)))
                    for i, user in enumerate(unique_users):
                        mask = np.array(user_ids) == user
                        axes[1, 0].scatter(latents_2d[mask, 0], latents_2d[mask, 1],
                                         c=[colors[i]], label=f'User {user}', alpha=0.6)
                    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                else:
                    axes[1, 0].scatter(latents_2d[:, 0], latents_2d[:, 1], alpha=0.6)

                axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
                axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
                axes[1, 0].set_title('Latent Space (PCA)')
            except:
                axes[1, 0].text(0.5, 0.5, 'PCA failed',
                              ha='center', va='center', transform=axes[1, 0].transAxes)
        else:
            axes[1, 0].text(0.5, 0.5, 'Insufficient data for PCA',
                          ha='center', va='center', transform=axes[1, 0].transAxes)

        # 4. 潜在向量的范数分布
        latent_norms = np.linalg.norm(latents, axis=1)
        axes[1, 1].hist(latent_norms, bins=50, alpha=0.7)
        axes[1, 1].set_xlabel('L2 Norm')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Latent Vector Norms')

        plt.tight_layout()
        plt.savefig(self.output_dir / "latent_space_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()

    def generate_comprehensive_report(self, reconstruction_results, codebook_results, latent_results):
        """生成综合评估报告"""
        print(f"\n📋 生成综合评估报告...")

        # 计算综合质量分数
        quality_score = self._calculate_quality_score(reconstruction_results, codebook_results, latent_results)

        # 生成文本报告
        report = self._generate_text_report(reconstruction_results, codebook_results, latent_results, quality_score)

        # 保存报告
        with open(self.output_dir / "comprehensive_report.txt", 'w', encoding='utf-8') as f:
            f.write(report)

        # 保存JSON结果
        results = {
            'reconstruction_quality': reconstruction_results,
            'codebook_quality': codebook_results,
            'latent_space_quality': latent_results,
            'overall_quality_score': quality_score,
        }

        with open(self.output_dir / "evaluation_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"📄 报告已保存到: {self.output_dir}")
        return quality_score

    def _calculate_quality_score(self, recon_results, codebook_results, latent_results):
        """计算综合质量分数 (0-100)"""
        scores = {}

        # 1. 重建质量分数 (40%)
        psnr_score = min(100, max(0, (recon_results['psnr']['mean'] - 20) * 2))  # 20dB=0分, 70dB=100分
        ssim_score = recon_results['ssim']['mean'] * 100  # 直接转换为百分制
        recon_score = (psnr_score + ssim_score) / 2
        scores['reconstruction'] = recon_score

        # 2. 码本质量分数 (40%)
        usage_score = codebook_results['usage_statistics']['usage_rate'] * 100
        entropy_score = codebook_results['usage_distribution']['normalized_entropy'] * 100
        consistency_score = 100 if codebook_results['reconstruction_consistency']['is_deterministic'] else 50
        codebook_score = (usage_score + entropy_score + consistency_score) / 3
        scores['codebook'] = codebook_score

        # 3. 潜在空间质量分数 (20%)
        sparsity_penalty = latent_results['statistical_properties']['sparsity'] * 50  # 稀疏性惩罚
        latent_score = max(0, 100 - sparsity_penalty)
        scores['latent_space'] = latent_score

        # 综合分数
        overall_score = (
            scores['reconstruction'] * 0.4 +
            scores['codebook'] * 0.4 +
            scores['latent_space'] * 0.2
        )

        scores['overall'] = overall_score
        return scores

    def _generate_text_report(self, recon_results, codebook_results, latent_results, quality_scores):
        """生成文本报告"""
        report = []
        report.append("=" * 80)
        report.append("VQ-VAE 质量全面评估报告")
        report.append("=" * 80)
        report.append("")

        # 总体评分
        overall_score = quality_scores['overall']
        if overall_score >= 80:
            grade = "优秀 (A)"
        elif overall_score >= 70:
            grade = "良好 (B)"
        elif overall_score >= 60:
            grade = "及格 (C)"
        else:
            grade = "需要改进 (D)"

        report.append(f"📊 总体质量评分: {overall_score:.1f}/100 - {grade}")
        report.append("")

        # 1. 重建质量评估
        report.append("🎯 1. 重建质量评估")
        report.append("-" * 40)
        psnr = recon_results['psnr']
        ssim = recon_results['ssim']
        report.append(f"PSNR: {psnr['mean']:.2f} ± {psnr['std']:.2f} dB (范围: {psnr['min']:.2f} - {psnr['max']:.2f})")
        report.append(f"SSIM: {ssim['mean']:.4f} ± {ssim['std']:.4f} (范围: {ssim['min']:.4f} - {ssim['max']:.4f})")
        report.append(f"MSE: {recon_results['mse']['mean']:.6f}")
        report.append(f"MAE: {recon_results['mae']['mean']:.6f}")

        if 'lpips' in recon_results:
            lpips = recon_results['lpips']
            report.append(f"LPIPS: {lpips['mean']:.4f} ± {lpips['std']:.4f}")

        # 重建质量判断
        if psnr['mean'] >= 30:
            report.append("✅ 重建质量: 优秀")
        elif psnr['mean'] >= 25:
            report.append("✅ 重建质量: 良好")
        elif psnr['mean'] >= 20:
            report.append("⚠️ 重建质量: 一般")
        else:
            report.append("❌ 重建质量: 较差")
        report.append("")

        # 2. 码本质量评估
        report.append("🎯 2. 码本质量评估")
        report.append("-" * 40)
        usage_stats = codebook_results['usage_statistics']
        usage_dist = codebook_results['usage_distribution']

        report.append(f"码本使用率: {usage_stats['usage_rate']:.3f} ({usage_stats['used_codes']}/{usage_stats['total_codes']})")
        report.append(f"使用熵: {usage_dist['entropy']:.3f} (归一化: {usage_dist['normalized_entropy']:.3f})")
        report.append(f"基尼系数: {usage_dist['gini_coefficient']:.3f}")

        consistency = codebook_results['reconstruction_consistency']
        report.append(f"重建一致性: {'确定性' if consistency['is_deterministic'] else '非确定性'}")

        # 码本质量判断
        usage_rate = usage_stats['usage_rate']
        if usage_rate >= 0.7:
            report.append("✅ 码本使用: 优秀，无坍缩风险")
        elif usage_rate >= 0.5:
            report.append("✅ 码本使用: 良好")
        elif usage_rate >= 0.3:
            report.append("⚠️ 码本使用: 一般，需要关注")
        else:
            report.append("❌ 码本使用: 较差，存在坍缩风险")
        report.append("")

        # 3. 潜在空间质量
        report.append("🎯 3. 潜在空间质量评估")
        report.append("-" * 40)
        stat_props = latent_results['statistical_properties']
        report.append(f"平均幅度: {stat_props['mean_magnitude']:.4f}")
        report.append(f"标准差幅度: {stat_props['std_magnitude']:.4f}")
        report.append(f"值域范围: {stat_props['range']:.4f}")
        report.append(f"稀疏性: {stat_props['sparsity']:.3f}")

        if latent_results['user_separation']:
            user_sep = latent_results['user_separation']
            report.append(f"用户分离比: {user_sep['separation_ratio']:.2f}")
            report.append(f"用户数量: {user_sep['num_users']}")

        report.append("")

        # 4. 建议和总结
        report.append("🎯 4. 改进建议")
        report.append("-" * 40)

        suggestions = []
        if psnr['mean'] < 25:
            suggestions.append("• 考虑增加模型容量或调整损失函数权重以提高重建质量")

        if usage_rate < 0.5:
            suggestions.append("• 码本使用率偏低，建议调整EMA衰减率或commitment权重")

        if stat_props['sparsity'] > 0.8:
            suggestions.append("• 潜在空间过于稀疏，可能影响表达能力")

        if not suggestions:
            suggestions.append("• 模型质量良好，可以考虑进一步优化超参数")

        for suggestion in suggestions:
            report.append(suggestion)

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)

    def run_comprehensive_evaluation(self):
        """运行完整的评估流程"""
        print(f"🚀 开始VQ-VAE全面质量评估...")

        # 加载模型
        model = self.load_model(self.args.model_path)

        # 创建数据加载器
        dataloader = self.create_dataloader()

        # 1. 重建质量评估
        reconstruction_results = self.evaluate_reconstruction_quality(model, dataloader)

        # 2. 码本质量评估
        codebook_results = self.evaluate_codebook_quality(model, dataloader)

        # 3. 潜在空间质量评估
        latent_results = self.evaluate_latent_space_quality(model, dataloader)

        # 4. 生成综合报告
        quality_scores = self.generate_comprehensive_report(
            reconstruction_results, codebook_results, latent_results
        )

        # 打印总结
        print(f"\n🎉 评估完成!")
        print(f"📊 总体质量分数: {quality_scores['overall']:.1f}/100")
        print(f"   - 重建质量: {quality_scores['reconstruction']:.1f}/100")
        print(f"   - 码本质量: {quality_scores['codebook']:.1f}/100")
        print(f"   - 潜在空间: {quality_scores['latent_space']:.1f}/100")
        print(f"📁 详细报告保存在: {self.output_dir}")

        return quality_scores


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="VQ-VAE质量全面评估")

    # 必需参数
    parser.add_argument("--model_path", type=str, required=True,
                       help="VQ-VAE模型路径 (checkpoint文件或模型目录)")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="验证数据集目录")

    # 可选参数
    parser.add_argument("--output_dir", type=str, default="outputs/vqvae_evaluation",
                       help="评估结果输出目录")
    parser.add_argument("--resolution", type=int, default=128,
                       help="图像分辨率")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="批次大小")
    parser.add_argument("--max_samples", type=int, default=1000,
                       help="最大评估样本数 (None表示使用全部)")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="数据加载器工作进程数")

    args = parser.parse_args()

    # 创建评估器并运行评估
    evaluator = ComprehensiveVQVAEEvaluator(args)
    quality_scores = evaluator.run_comprehensive_evaluation()

    # 根据分数给出最终建议
    overall_score = quality_scores['overall']
    print(f"\n🎯 最终评估结论:")

    if overall_score >= 80:
        print(f"🏆 模型质量优秀! 可以用于生产环境")
    elif overall_score >= 70:
        print(f"✅ 模型质量良好，可以考虑部署")
    elif overall_score >= 60:
        print(f"⚠️ 模型质量一般，建议进一步优化")
    else:
        print(f"❌ 模型质量较差，需要重新训练或调整架构")

    print(f"📋 详细分析报告请查看: {args.output_dir}/comprehensive_report.txt")


if __name__ == "__main__":
    main()
