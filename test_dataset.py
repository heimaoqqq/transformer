#!/usr/bin/env python3
"""
数据集验证脚本
验证数据集格式、图像尺寸、归一化等
"""

import os
import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from utils.data_loader import MicroDopplerDataset, get_default_transform, denormalize_tensor, tensor_to_pil

def test_dataset_structure(data_dir: str):
    """测试数据集结构"""
    print("📁 数据集结构检查:")
    print(f"   数据目录: {data_dir}")
    
    if not os.path.exists(data_dir):
        print("❌ 数据集目录不存在")
        return False
    
    data_path = Path(data_dir)
    user_dirs = []
    
    # 查找用户目录
    for item in data_path.iterdir():
        if item.is_dir() and item.name.startswith('ID'):
            user_dirs.append(item)
    
    user_dirs.sort(key=lambda x: x.name)
    
    print(f"✅ 找到 {len(user_dirs)} 个用户目录")
    
    total_images = 0
    user_stats = {}
    
    for user_dir in user_dirs:
        # 解析用户ID
        dir_name = user_dir.name
        try:
            if '_' in dir_name:
                user_id = int(dir_name.split('_')[1])  # ID_2 -> 2
            else:
                user_id = int(dir_name[2:])  # ID1 -> 1
        except ValueError:
            print(f"⚠️ 无法解析用户ID: {dir_name}")
            continue
        
        # 统计图像文件
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(list(user_dir.glob(ext)))
        
        user_stats[user_id] = len(image_files)
        total_images += len(image_files)
        
        if len(user_dirs) <= 10:  # 如果用户不多，显示详细信息
            print(f"   {dir_name} (用户{user_id}): {len(image_files)} 张图像")
    
    if len(user_dirs) > 10:
        print(f"   用户ID范围: {min(user_stats.keys())} - {max(user_stats.keys())}")
        print(f"   平均每用户: {total_images / len(user_stats):.1f} 张图像")
    
    print(f"✅ 总计: {total_images} 张图像")
    return True, user_stats

def test_image_properties(data_dir: str, sample_count: int = 5):
    """测试图像属性"""
    print("\n🖼️ 图像属性检查:")
    
    data_path = Path(data_dir)
    sample_images = []
    
    # 收集样本图像
    for user_dir in data_path.iterdir():
        if user_dir.is_dir() and user_dir.name.startswith('ID'):
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                for img_path in user_dir.glob(ext):
                    sample_images.append(img_path)
                    if len(sample_images) >= sample_count:
                        break
                if len(sample_images) >= sample_count:
                    break
            if len(sample_images) >= sample_count:
                break
    
    if not sample_images:
        print("❌ 未找到图像文件")
        return False
    
    print(f"   检查 {len(sample_images)} 张样本图像...")
    
    sizes = []
    modes = []
    
    for img_path in sample_images:
        try:
            with Image.open(img_path) as img:
                sizes.append(img.size)
                modes.append(img.mode)
        except Exception as e:
            print(f"⚠️ 无法读取图像 {img_path}: {e}")
    
    # 统计尺寸
    unique_sizes = list(set(sizes))
    print(f"   图像尺寸: {unique_sizes}")
    
    if len(unique_sizes) == 1 and unique_sizes[0] == (256, 256):
        print("✅ 所有图像都是256x256")
    else:
        print("⚠️ 图像尺寸不一致或不是256x256")
    
    # 统计模式
    unique_modes = list(set(modes))
    print(f"   颜色模式: {unique_modes}")
    
    if len(unique_modes) == 1 and unique_modes[0] == 'RGB':
        print("✅ 所有图像都是RGB模式")
    else:
        print("⚠️ 图像颜色模式不一致或不是RGB")
    
    return True

def test_data_loading(data_dir: str):
    """测试数据加载"""
    print("\n🔄 数据加载测试:")
    
    try:
        # 创建数据集
        transform = get_default_transform(resolution=128, normalize=True)
        dataset = MicroDopplerDataset(
            data_dir=data_dir,
            transform=transform,
            return_user_id=True,
        )
        
        print(f"✅ 数据集创建成功")
        print(f"   总样本数: {len(dataset)}")
        
        # 获取用户统计
        user_stats = dataset.get_user_statistics()
        print(f"   用户数量: {len(user_stats)}")
        print(f"   用户ID范围: {min(user_stats.keys())} - {max(user_stats.keys())}")
        
        # 测试数据加载
        if len(dataset) > 0:
            image, user_id = dataset[0]
            print(f"✅ 数据加载成功")
            print(f"   图像形状: {image.shape}")
            print(f"   图像范围: [{image.min():.3f}, {image.max():.3f}]")
            print(f"   用户ID: {user_id}")
            
            # 测试反归一化
            denorm_image = denormalize_tensor(image)
            print(f"   反归一化后范围: [{denorm_image.min():.3f}, {denorm_image.max():.3f}]")
            
            return True
        else:
            print("❌ 数据集为空")
            return False
            
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return False

def main():
    """主函数"""
    data_dir = "/kaggle/input/dataset"
    
    print("🧪 数据集验证测试")
    print("=" * 50)
    
    # 1. 结构检查
    success, user_stats = test_dataset_structure(data_dir)
    if not success:
        return
    
    # 2. 图像属性检查
    test_image_properties(data_dir)
    
    # 3. 数据加载测试
    test_data_loading(data_dir)
    
    print("\n🎉 数据集验证完成!")
    print("\n💡 使用建议:")
    print("   - 确保所有图像都是256x256 RGB格式")
    print("   - 用户目录命名: ID1, ID_2, ID_3, ..., ID_31")
    print("   - 训练时会自动缩放到128x128")
    print("   - 归一化范围: [-1, 1]")

if __name__ == "__main__":
    main()
