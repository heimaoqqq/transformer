#!/usr/bin/env python3
"""
微多普勒图像生成质量评估指标
包括FID、LPIPS、PSNR、SSIM等指标
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Union, Optional
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy import linalg
import torchvision.transforms as transforms

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: LPIPS not available. Install with: pip install lpips")

try:
    from pytorch_fid import fid_score
    from pytorch_fid.inception import InceptionV3
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False
    print("Warning: pytorch-fid not available. Install with: pip install pytorch-fid")

class MetricsCalculator:
    """图像生成质量评估指标计算器"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        
        # 初始化LPIPS
        if LPIPS_AVAILABLE:
            self.lpips_model = lpips.LPIPS(net='alex').to(device)
            self.lpips_model.eval()
        else:
            self.lpips_model = None
        
        # 初始化FID模型
        if FID_AVAILABLE:
            self.inception_model = InceptionV3().to(device)
            self.inception_model.eval()
        else:
            self.inception_model = None
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),  # Inception输入尺寸
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def calculate_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        计算PSNR (Peak Signal-to-Noise Ratio)
        
        Args:
            img1: 第一张图像 [H, W, C] 或 [H, W]
            img2: 第二张图像 [H, W, C] 或 [H, W]
            
        Returns:
            PSNR值
        """
        return psnr(img1, img2, data_range=255)
    
    def calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        计算SSIM (Structural Similarity Index)
        
        Args:
            img1: 第一张图像 [H, W, C] 或 [H, W]
            img2: 第二张图像 [H, W, C] 或 [H, W]
            
        Returns:
            SSIM值
        """
        if len(img1.shape) == 3:
            # 多通道图像
            return ssim(img1, img2, multichannel=True, data_range=255, channel_axis=2)
        else:
            # 单通道图像
            return ssim(img1, img2, data_range=255)
    
    def calculate_lpips(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """
        计算LPIPS (Learned Perceptual Image Patch Similarity)
        
        Args:
            img1: 第一张图像 [B, C, H, W] 或 [C, H, W]
            img2: 第二张图像 [B, C, H, W] 或 [C, H, W]
            
        Returns:
            LPIPS值
        """
        if self.lpips_model is None:
            raise ValueError("LPIPS model not available")
        
        # 确保输入是4D张量
        if len(img1.shape) == 3:
            img1 = img1.unsqueeze(0)
        if len(img2.shape) == 3:
            img2 = img2.unsqueeze(0)
        
        # 归一化到[-1, 1]
        img1 = img1 * 2.0 - 1.0
        img2 = img2 * 2.0 - 1.0
        
        # 移到设备
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)
        
        with torch.no_grad():
            distance = self.lpips_model(img1, img2)
        
        return distance.mean().item()
    
    def calculate_fid(self, real_images: List[np.ndarray], fake_images: List[np.ndarray]) -> float:
        """
        计算FID (Fréchet Inception Distance)
        
        Args:
            real_images: 真实图像列表
            fake_images: 生成图像列表
            
        Returns:
            FID值
        """
        if self.inception_model is None:
            raise ValueError("Inception model not available")
        
        # 提取特征
        real_features = self._extract_inception_features(real_images)
        fake_features = self._extract_inception_features(fake_images)
        
        # 计算FID
        fid_value = self._calculate_fid_from_features(real_features, fake_features)
        
        return fid_value
    
    def _extract_inception_features(self, images: List[np.ndarray]) -> np.ndarray:
        """提取Inception特征"""
        features = []
        
        with torch.no_grad():
            for img in images:
                # 转换为PIL图像
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8)
                
                pil_img = Image.fromarray(img)
                
                # 预处理
                tensor_img = self.transform(pil_img).unsqueeze(0).to(self.device)
                
                # 提取特征
                feature = self.inception_model(tensor_img)[0]
                feature = feature.cpu().numpy().flatten()
                features.append(feature)
        
        return np.array(features)
    
    def _calculate_fid_from_features(self, real_features: np.ndarray, fake_features: np.ndarray) -> float:
        """从特征计算FID"""
        # 计算均值和协方差
        mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
        
        # 计算FID
        diff = mu1 - mu2
        
        # 计算协方差矩阵的平方根
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % 1e-6
            print(msg)
            offset = np.eye(sigma1.shape[0]) * 1e-6
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        # 数值稳定性
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real
        
        tr_covmean = np.trace(covmean)
        
        return (diff.dot(diff) + np.trace(sigma1) + 
                np.trace(sigma2) - 2 * tr_covmean)
    
    def calculate_frequency_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        计算频域相似性 (针对时频图的特殊指标)
        
        Args:
            img1: 第一张图像
            img2: 第二张图像
            
        Returns:
            频域相似性分数
        """
        # 转换为灰度图
        if len(img1.shape) == 3:
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        else:
            img1_gray = img1
            img2_gray = img2
        
        # FFT变换
        fft1 = np.fft.fft2(img1_gray)
        fft2 = np.fft.fft2(img2_gray)
        
        # 计算幅度谱
        mag1 = np.abs(fft1)
        mag2 = np.abs(fft2)
        
        # 归一化
        mag1 = mag1 / np.max(mag1)
        mag2 = mag2 / np.max(mag2)
        
        # 计算相似性 (使用余弦相似性)
        mag1_flat = mag1.flatten()
        mag2_flat = mag2.flatten()
        
        similarity = np.dot(mag1_flat, mag2_flat) / (
            np.linalg.norm(mag1_flat) * np.linalg.norm(mag2_flat)
        )
        
        return similarity
    
    def evaluate_reconstruction(
        self, 
        original_images: List[np.ndarray], 
        reconstructed_images: List[np.ndarray]
    ) -> dict:
        """
        评估重建质量
        
        Args:
            original_images: 原始图像列表
            reconstructed_images: 重建图像列表
            
        Returns:
            评估指标字典
        """
        if len(original_images) != len(reconstructed_images):
            raise ValueError("Image lists must have the same length")
        
        psnr_scores = []
        ssim_scores = []
        lpips_scores = []
        freq_similarities = []
        
        for orig, recon in zip(original_images, reconstructed_images):
            # PSNR和SSIM
            psnr_scores.append(self.calculate_psnr(orig, recon))
            ssim_scores.append(self.calculate_ssim(orig, recon))
            
            # 频域相似性
            freq_similarities.append(self.calculate_frequency_similarity(orig, recon))
            
            # LPIPS (如果可用)
            if self.lpips_model is not None:
                # 转换为tensor
                orig_tensor = torch.from_numpy(orig).permute(2, 0, 1).float() / 255.0
                recon_tensor = torch.from_numpy(recon).permute(2, 0, 1).float() / 255.0
                
                lpips_scores.append(self.calculate_lpips(orig_tensor, recon_tensor))
        
        results = {
            'psnr_mean': np.mean(psnr_scores),
            'psnr_std': np.std(psnr_scores),
            'ssim_mean': np.mean(ssim_scores),
            'ssim_std': np.std(ssim_scores),
            'freq_similarity_mean': np.mean(freq_similarities),
            'freq_similarity_std': np.std(freq_similarities),
        }
        
        if lpips_scores:
            results.update({
                'lpips_mean': np.mean(lpips_scores),
                'lpips_std': np.std(lpips_scores),
            })
        
        return results
    
    def evaluate_generation(
        self,
        real_images: List[np.ndarray],
        generated_images: List[np.ndarray]
    ) -> dict:
        """
        评估生成质量
        
        Args:
            real_images: 真实图像列表
            generated_images: 生成图像列表
            
        Returns:
            评估指标字典
        """
        results = {}
        
        # FID
        if self.inception_model is not None:
            try:
                fid_value = self.calculate_fid(real_images, generated_images)
                results['fid'] = fid_value
            except Exception as e:
                print(f"FID calculation failed: {e}")
        
        # 图像质量统计
        real_stats = self._calculate_image_statistics(real_images)
        gen_stats = self._calculate_image_statistics(generated_images)
        
        results.update({
            'real_mean_brightness': real_stats['mean_brightness'],
            'gen_mean_brightness': gen_stats['mean_brightness'],
            'real_std_brightness': real_stats['std_brightness'],
            'gen_std_brightness': gen_stats['std_brightness'],
            'brightness_diff': abs(real_stats['mean_brightness'] - gen_stats['mean_brightness']),
        })
        
        return results
    
    def _calculate_image_statistics(self, images: List[np.ndarray]) -> dict:
        """计算图像统计信息"""
        brightnesses = []
        
        for img in images:
            # 转换为灰度图计算亮度
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img
            
            brightness = np.mean(gray)
            brightnesses.append(brightness)
        
        return {
            'mean_brightness': np.mean(brightnesses),
            'std_brightness': np.std(brightnesses),
        }

# 独立函数，用于训练脚本
def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    计算PSNR (Peak Signal-to-Noise Ratio)

    Args:
        img1: 第一张图像 [H, W, C] 或 [H, W]
        img2: 第二张图像 [H, W, C] 或 [H, W]

    Returns:
        PSNR值
    """
    return psnr(img1, img2, data_range=255)

def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    计算SSIM (Structural Similarity Index)

    Args:
        img1: 第一张图像 [H, W, C] 或 [H, W]
        img2: 第二张图像 [H, W, C] 或 [H, W]

    Returns:
        SSIM值
    """
    if len(img1.shape) == 3:
        # 多通道图像
        return ssim(img1, img2, multichannel=True, data_range=255, channel_axis=2)
    else:
        # 单通道图像
        return ssim(img1, img2, data_range=255)

def load_images_from_directory(directory: Union[str, Path]) -> List[np.ndarray]:
    """从目录加载图像"""
    directory = Path(directory)
    images = []

    for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
        for img_path in directory.glob(ext):
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img)
            images.append(img_array)

    return images

def evaluate_model_performance(
    real_data_dir: str,
    generated_data_dir: str,
    device: str = "cuda"
) -> dict:
    """
    评估模型性能
    
    Args:
        real_data_dir: 真实数据目录
        generated_data_dir: 生成数据目录
        device: 计算设备
        
    Returns:
        评估结果字典
    """
    # 初始化计算器
    calculator = MetricsCalculator(device=device)
    
    # 加载图像
    print("Loading real images...")
    real_images = load_images_from_directory(real_data_dir)
    
    print("Loading generated images...")
    generated_images = load_images_from_directory(generated_data_dir)
    
    print(f"Loaded {len(real_images)} real images and {len(generated_images)} generated images")
    
    # 评估生成质量
    print("Calculating generation metrics...")
    results = calculator.evaluate_generation(real_images, generated_images)
    
    return results

if __name__ == "__main__":
    # 测试代码
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Model Performance")
    parser.add_argument("--real_dir", type=str, required=True, help="真实数据目录")
    parser.add_argument("--generated_dir", type=str, required=True, help="生成数据目录")
    parser.add_argument("--device", type=str, default="cuda", help="计算设备")
    
    args = parser.parse_args()
    
    results = evaluate_model_performance(
        real_data_dir=args.real_dir,
        generated_data_dir=args.generated_dir,
        device=args.device
    )
    
    print("\nEvaluation Results:")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")
