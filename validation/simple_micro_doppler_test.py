#!/usr/bin/env python3
"""
简化版微多普勒分析工具 - 避免复杂的数组操作
专门用于快速测试和诊断
"""

import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

def find_user_directory(data_root: str, user_id: int):
    """查找用户目录，支持多种格式"""
    data_path = Path(data_root)
    
    # 支持的目录格式
    possible_names = [
        f"user_{user_id:02d}",  # user_01
        f"user_{user_id}",      # user_1
        f"ID_{user_id}",        # ID_1
        f"{user_id}"            # 1
    ]
    
    for name in possible_names:
        user_dir = data_path / name
        if user_dir.exists() and user_dir.is_dir():
            return user_dir
    
    return None

def simple_load_images(user_dir: Path, max_samples: int = 5):
    """简单加载图像，避免复杂操作"""
    print(f"  📂 检查目录: {user_dir}")
    
    if not user_dir.exists():
        print(f"    ❌ 目录不存在")
        return []
    
    # 查找图像文件
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(list(user_dir.glob(ext)))
    
    print(f"    📁 找到 {len(image_files)} 个图像文件")
    
    if not image_files:
        return []
    
    images = []
    for i, img_file in enumerate(image_files[:max_samples]):
        try:
            print(f"    📷 加载图像 {i+1}: {img_file.name}")
            img = Image.open(img_file)
            print(f"      原始尺寸: {img.size}, 模式: {img.mode}")
            
            # 转换为RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 调整尺寸
            img = img.resize((128, 128))
            
            # 转换为数组
            img_array = np.array(img)
            print(f"      数组形状: {img_array.shape}, 数据类型: {img_array.dtype}")
            print(f"      像素值范围: [{img_array.min()}, {img_array.max()}]")
            
            # 归一化
            img_array = img_array.astype(np.float32) / 255.0
            images.append(img_array)
            
        except Exception as e:
            print(f"    ❌ 加载失败: {e}")
    
    print(f"    ✅ 成功加载 {len(images)} 张图像")
    return images

def simple_analyze_differences(data_root: str, user_ids: list = [1, 2]):
    """简单分析用户间差异"""
    print("🔍 简单用户差异分析")
    print("=" * 50)
    
    # 加载每个用户的图像
    user_data = {}
    for user_id in user_ids:
        print(f"\n👤 分析用户 {user_id}")
        
        user_dir = find_user_directory(data_root, user_id)
        if not user_dir:
            print(f"  ❌ 未找到用户 {user_id} 的目录")
            continue
        
        images = simple_load_images(user_dir, max_samples=3)
        if not images:
            print(f"  ❌ 未能加载用户 {user_id} 的图像")
            continue
        
        # 计算简单统计
        images_array = np.array(images)
        print(f"  📊 图像数组形状: {images_array.shape}")
        
        # 计算平均图像
        mean_img = np.mean(images_array, axis=0)
        print(f"  📊 平均图像形状: {mean_img.shape}")
        
        # 转换为灰度（如果是RGB）
        if len(mean_img.shape) == 3 and mean_img.shape[2] == 3:
            gray_img = np.mean(mean_img, axis=2)
            print(f"  📊 灰度图像形状: {gray_img.shape}")
        else:
            gray_img = mean_img
        
        # 计算基本统计
        stats = {
            'mean_brightness': np.mean(gray_img),
            'std_brightness': np.std(gray_img),
            'min_value': np.min(gray_img),
            'max_value': np.max(gray_img),
            'image_shape': gray_img.shape
        }
        
        print(f"  📈 统计信息:")
        print(f"    平均亮度: {stats['mean_brightness']:.4f}")
        print(f"    亮度标准差: {stats['std_brightness']:.4f}")
        print(f"    值域: [{stats['min_value']:.4f}, {stats['max_value']:.4f}]")
        
        user_data[user_id] = {
            'images': images,
            'mean_img': mean_img,
            'gray_img': gray_img,
            'stats': stats
        }
    
    # 比较用户间差异
    if len(user_data) >= 2:
        print(f"\n🔍 用户间差异比较")
        print("=" * 30)
        
        user_list = list(user_data.keys())
        for i in range(len(user_list)):
            for j in range(i + 1, len(user_list)):
                user1, user2 = user_list[i], user_list[j]
                
                print(f"\n👥 用户 {user1} vs 用户 {user2}:")
                
                stats1 = user_data[user1]['stats']
                stats2 = user_data[user2]['stats']
                
                # 亮度差异
                brightness_diff = abs(stats1['mean_brightness'] - stats2['mean_brightness'])
                print(f"  亮度差异: {brightness_diff:.4f}")
                
                # 对比度差异
                contrast_diff = abs(stats1['std_brightness'] - stats2['std_brightness'])
                print(f"  对比度差异: {contrast_diff:.4f}")
                
                # 像素级差异
                try:
                    gray1 = user_data[user1]['gray_img']
                    gray2 = user_data[user2]['gray_img']
                    
                    if gray1.shape == gray2.shape:
                        pixel_diff = np.mean(np.abs(gray1 - gray2))
                        print(f"  像素级差异: {pixel_diff:.4f}")
                        
                        # 判断差异程度
                        if pixel_diff < 0.01:
                            print(f"  🚨 差异极小，几乎相同")
                        elif pixel_diff < 0.05:
                            print(f"  ⚠️  差异较小，需要强化")
                        elif pixel_diff < 0.1:
                            print(f"  📊 差异中等，可以区分")
                        else:
                            print(f"  ✅ 差异明显，容易区分")
                    else:
                        print(f"  ❌ 图像尺寸不匹配: {gray1.shape} vs {gray2.shape}")
                        
                except Exception as e:
                    print(f"  ❌ 比较失败: {e}")
    
    return user_data

def test_basic_functionality(data_root: str):
    """测试基本功能"""
    print("🧪 基本功能测试")
    print("=" * 50)
    
    data_path = Path(data_root)
    print(f"📂 数据根目录: {data_path}")
    print(f"   存在: {data_path.exists()}")
    
    if not data_path.exists():
        print("❌ 数据目录不存在")
        return False
    
    # 列出所有子目录
    subdirs = [d for d in data_path.iterdir() if d.is_dir()]
    print(f"📁 找到 {len(subdirs)} 个子目录:")
    for subdir in subdirs[:10]:  # 只显示前10个
        print(f"   {subdir.name}")
    
    # 尝试识别用户目录
    user_dirs = []
    for subdir in subdirs:
        if any(pattern in subdir.name for pattern in ['user_', 'ID_']):
            user_dirs.append(subdir)
    
    print(f"🔍 识别出 {len(user_dirs)} 个用户目录:")
    for user_dir in user_dirs[:5]:  # 只显示前5个
        print(f"   {user_dir.name}")
    
    return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="简化版微多普勒分析")
    parser.add_argument("--data_root", type=str, required=True, help="数据根目录")
    parser.add_argument("--user_ids", type=int, nargs='+', default=[1, 2], help="用户ID列表")
    parser.add_argument("--test_basic", action="store_true", help="只测试基本功能")
    
    args = parser.parse_args()
    
    print("🔬 简化版微多普勒分析工具")
    print("=" * 60)
    
    # 测试基本功能
    if not test_basic_functionality(args.data_root):
        return
    
    if args.test_basic:
        print("✅ 基本功能测试完成")
        return
    
    # 分析用户差异
    try:
        user_data = simple_analyze_differences(args.data_root, args.user_ids)
        
        if user_data:
            print(f"\n✅ 分析完成，成功处理 {len(user_data)} 个用户")
        else:
            print(f"\n❌ 分析失败，未能处理任何用户")
            
    except Exception as e:
        print(f"\n❌ 分析过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
