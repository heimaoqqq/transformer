#!/usr/bin/env python3
"""
图像插值质量对比测试
比较不同插值方法的效果
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

def test_interpolation_methods(image_path: str, target_size: int = 128):
    """
    测试不同插值方法的效果
    Args:
        image_path: 测试图像路径
        target_size: 目标尺寸
    """
    if not os.path.exists(image_path):
        print(f"❌ 图像文件不存在: {image_path}")
        return
    
    # 加载原始图像
    original_image = Image.open(image_path).convert('RGB')
    print(f"📷 原始图像尺寸: {original_image.size}")
    
    # 定义插值方法
    methods = {
        'Bilinear': transforms.InterpolationMode.BILINEAR,
        'Bicubic': transforms.InterpolationMode.BICUBIC,
        'Lanczos': transforms.InterpolationMode.LANCZOS,
    }
    
    # 创建变换
    transforms_dict = {}
    
    for name, mode in methods.items():
        transforms_dict[name] = transforms.Compose([
            transforms.Resize((target_size, target_size), interpolation=mode),
            transforms.ToTensor(),
        ])
    
    # 添加抗锯齿方法 (如果支持)
    try:
        transforms_dict['Antialias'] = transforms.Compose([
            transforms.Resize((target_size, target_size), 
                            interpolation=transforms.InterpolationMode.BILINEAR,
                            antialias=True),
            transforms.ToTensor(),
        ])
    except TypeError:
        print("⚠️ 当前PyTorch版本不支持antialias参数")
    
    # 应用变换
    results = {}
    for name, transform in transforms_dict.items():
        tensor = transform(original_image)
        results[name] = tensor
    
    # 可视化结果
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # 显示原始图像
    axes[0].imshow(original_image)
    axes[0].set_title(f'Original ({original_image.size[0]}x{original_image.size[1]})')
    axes[0].axis('off')
    
    # 显示缩放结果
    for i, (name, tensor) in enumerate(results.items(), 1):
        if i < len(axes):
            # 转换为numpy数组显示
            img_array = tensor.permute(1, 2, 0).numpy()
            axes[i].imshow(img_array)
            axes[i].set_title(f'{name} ({target_size}x{target_size})')
            axes[i].axis('off')
    
    # 隐藏多余的子图
    for i in range(len(results) + 1, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('interpolation_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return results

def calculate_image_metrics(tensor1: torch.Tensor, tensor2: torch.Tensor):
    """
    计算图像质量指标
    Args:
        tensor1: 参考图像tensor
        tensor2: 比较图像tensor
    Returns:
        metrics: 质量指标字典
    """
    # 确保tensor在相同设备和数据类型
    tensor1 = tensor1.float()
    tensor2 = tensor2.float()
    
    # MSE (均方误差)
    mse = torch.mean((tensor1 - tensor2) ** 2).item()
    
    # PSNR (峰值信噪比)
    if mse > 0:
        psnr = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(torch.tensor(mse))
        psnr = psnr.item()
    else:
        psnr = float('inf')
    
    return {'mse': mse, 'psnr': psnr}

def benchmark_interpolation_speed(image_path: str, target_size: int = 128, iterations: int = 100):
    """
    测试插值方法的速度
    Args:
        image_path: 测试图像路径
        target_size: 目标尺寸
        iterations: 测试迭代次数
    """
    import time
    
    if not os.path.exists(image_path):
        print(f"❌ 图像文件不存在: {image_path}")
        return
    
    original_image = Image.open(image_path).convert('RGB')
    
    methods = {
        'Bilinear': transforms.InterpolationMode.BILINEAR,
        'Bicubic': transforms.InterpolationMode.BICUBIC,
        'Lanczos': transforms.InterpolationMode.LANCZOS,
    }
    
    print(f"⏱️ 速度测试 ({iterations} 次迭代):")
    
    for name, mode in methods.items():
        transform = transforms.Compose([
            transforms.Resize((target_size, target_size), interpolation=mode),
            transforms.ToTensor(),
        ])
        
        # 预热
        for _ in range(10):
            _ = transform(original_image)
        
        # 计时
        start_time = time.time()
        for _ in range(iterations):
            _ = transform(original_image)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / iterations * 1000  # 毫秒
        print(f"   {name:10s}: {avg_time:.2f} ms/image")

def find_sample_image(data_dir: str = "/kaggle/input/dataset"):
    """
    查找样本图像用于测试
    Args:
        data_dir: 数据集目录
    Returns:
        sample_path: 样本图像路径
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        return None
    
    # 查找第一个图像文件
    for user_dir in data_path.iterdir():
        if user_dir.is_dir() and user_dir.name.startswith('ID'):
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                image_files = list(user_dir.glob(ext))
                if image_files:
                    return str(image_files[0])
    
    print("❌ 未找到图像文件")
    return None

def main():
    """主函数"""
    print("🔍 图像插值质量对比测试")
    print("=" * 50)
    
    # 查找样本图像
    sample_image = find_sample_image()
    
    if sample_image is None:
        print("💡 请提供测试图像路径")
        return
    
    print(f"📷 使用样本图像: {sample_image}")
    
    # 1. 质量对比
    print("\n1. 插值质量对比:")
    results = test_interpolation_methods(sample_image, target_size=128)
    
    # 2. 速度测试
    print("\n2. 插值速度测试:")
    benchmark_interpolation_speed(sample_image, target_size=128, iterations=50)
    
    # 3. 推荐建议
    print("\n💡 推荐建议:")
    print("   🏆 最佳质量: Lanczos (细节保持最好)")
    print("   ⚡ 最佳速度: Bilinear (最快)")
    print("   🎯 平衡选择: Bicubic (质量和速度平衡)")
    print("   🆕 现代方法: Antialias (减少锯齿)")
    print("\n📊 对于微多普勒时频图:")
    print("   - 推荐使用 Lanczos 保持频谱细节")
    print("   - 如果速度重要可选择 Bicubic")
    print("   - 避免 Bilinear (会模糊重要特征)")

if __name__ == "__main__":
    main()
