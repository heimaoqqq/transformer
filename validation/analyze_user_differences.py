#!/usr/bin/env python3
"""
专门分析用户间差异的工具
基于你展示的热力图数据，分析为什么验证失败
"""

import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import re

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

def load_and_analyze_images(user_dir: Path, max_samples: int = 20):
    """加载并分析用户图像"""
    image_files = list(user_dir.glob("*.png")) + list(user_dir.glob("*.jpg"))

    if not image_files:
        print(f"⚠️  在目录 {user_dir} 中未找到图像文件")
        return np.array([])

    images = []
    for img_file in image_files[:max_samples]:
        try:
            img = Image.open(img_file).convert('RGB')
            img = img.resize((128, 128))  # 标准化尺寸
            img_array = np.array(img) / 255.0  # 归一化到[0,1]
            images.append(img_array)
        except Exception as e:
            print(f"⚠️  加载图像失败 {img_file}: {e}")

    if not images:
        print(f"⚠️  未能成功加载任何图像")
        return np.array([])

    return np.array(images)

def calculate_image_statistics(images):
    """计算图像统计信息"""
    if len(images) == 0:
        return None
    
    # 基本统计
    mean_img = np.mean(images, axis=0)
    std_img = np.std(images, axis=0)
    
    # 颜色分布统计
    mean_rgb = np.mean(mean_img, axis=(0, 1))  # 平均RGB值
    std_rgb = np.mean(std_img, axis=(0, 1))    # RGB标准差
    
    # 亮度统计
    brightness = np.mean(mean_img)
    contrast = np.std(mean_img)
    
    # 边缘密度（简单的梯度统计）
    grad_x = np.abs(np.diff(mean_img, axis=1))
    grad_y = np.abs(np.diff(mean_img, axis=0))
    edge_density = np.mean(grad_x) + np.mean(grad_y)
    
    return {
        'mean_img': mean_img,
        'std_img': std_img,
        'mean_rgb': mean_rgb,
        'std_rgb': std_rgb,
        'brightness': brightness,
        'contrast': contrast,
        'edge_density': edge_density,
        'num_images': len(images)
    }

def compare_users(user1_stats, user2_stats, user1_id, user2_id):
    """比较两个用户的差异"""
    print(f"\n🔍 用户 {user1_id} vs 用户 {user2_id} 详细比较:")
    
    # 图像级差异
    img_diff = np.mean(np.abs(user1_stats['mean_img'] - user2_stats['mean_img']))
    max_img_diff = np.max(np.abs(user1_stats['mean_img'] - user2_stats['mean_img']))
    
    # RGB差异
    rgb_diff = np.linalg.norm(user1_stats['mean_rgb'] - user2_stats['mean_rgb'])
    
    # 亮度和对比度差异
    brightness_diff = abs(user1_stats['brightness'] - user2_stats['brightness'])
    contrast_diff = abs(user1_stats['contrast'] - user2_stats['contrast'])
    
    # 边缘密度差异
    edge_diff = abs(user1_stats['edge_density'] - user2_stats['edge_density'])
    
    print(f"  📊 像素级差异:")
    print(f"    平均绝对差异: {img_diff:.4f}")
    print(f"    最大像素差异: {max_img_diff:.4f}")
    
    print(f"  🎨 颜色差异:")
    print(f"    RGB向量距离: {rgb_diff:.4f}")
    print(f"    用户{user1_id} RGB: [{user1_stats['mean_rgb'][0]:.3f}, {user1_stats['mean_rgb'][1]:.3f}, {user1_stats['mean_rgb'][2]:.3f}]")
    print(f"    用户{user2_id} RGB: [{user2_stats['mean_rgb'][0]:.3f}, {user2_stats['mean_rgb'][1]:.3f}, {user2_stats['mean_rgb'][2]:.3f}]")
    
    print(f"  💡 视觉特征差异:")
    print(f"    亮度差异: {brightness_diff:.4f}")
    print(f"    对比度差异: {contrast_diff:.4f}")
    print(f"    边缘密度差异: {edge_diff:.4f}")
    
    # 计算综合差异分数
    # 对于热力图数据，颜色和亮度差异更重要
    composite_score = (
        img_diff * 2.0 +           # 像素差异权重2
        rgb_diff * 1.5 +           # 颜色差异权重1.5
        brightness_diff * 1.0 +    # 亮度差异权重1
        contrast_diff * 0.5 +      # 对比度差异权重0.5
        edge_diff * 0.5            # 边缘差异权重0.5
    ) / 5.5
    
    print(f"  🎯 综合差异分数: {composite_score:.4f}")
    
    # 判断差异程度（针对热力图数据调整阈值）
    if composite_score < 0.02:
        print(f"    🚨 差异极小：扩散模型几乎不可能学到区别")
        difficulty = "极难"
    elif composite_score < 0.05:
        print(f"    ❌ 差异很小：扩散模型很难学到明显特征")
        difficulty = "很难"
    elif composite_score < 0.1:
        print(f"    ⚠️  差异较小：需要强化训练和高指导强度")
        difficulty = "较难"
    elif composite_score < 0.2:
        print(f"    📊 差异中等：应该可以学到一些特征")
        difficulty = "中等"
    else:
        print(f"    ✅ 差异明显：应该能学到明显特征")
        difficulty = "容易"
    
    return {
        'img_diff': img_diff,
        'rgb_diff': rgb_diff,
        'brightness_diff': brightness_diff,
        'contrast_diff': contrast_diff,
        'edge_diff': edge_diff,
        'composite_score': composite_score,
        'difficulty': difficulty
    }

def analyze_user_differences(data_root: str, target_user_ids: list = None, max_samples: int = 20):
    """分析多个用户间的差异"""
    print(f"🔍 分析用户间差异")
    print(f"数据根目录: {data_root}")
    print(f"=" * 60)
    
    data_path = Path(data_root)
    if not data_path.exists():
        print(f"❌ 数据目录不存在: {data_root}")
        return False
    
    # 自动发现所有用户或使用指定用户
    if target_user_ids is None:
        target_user_ids = []
        for item in data_path.iterdir():
            if item.is_dir():
                # 尝试提取用户ID
                for pattern in [r'user_(\d+)', r'ID_(\d+)', r'^(\d+)$']:
                    match = re.match(pattern, item.name)
                    if match:
                        user_id = int(match.group(1))
                        target_user_ids.append(user_id)
                        break
        target_user_ids = sorted(list(set(target_user_ids)))
    
    print(f"📁 发现用户: {target_user_ids}")
    
    # 加载所有用户数据
    user_stats = {}
    for user_id in target_user_ids:
        user_dir = find_user_directory(data_root, user_id)
        if user_dir:
            print(f"\n📂 加载用户 {user_id} 数据...")
            images = load_and_analyze_images(user_dir, max_samples)
            if len(images) > 0:
                stats = calculate_image_statistics(images)
                user_stats[user_id] = stats
                print(f"  ✅ 成功加载 {len(images)} 张图像")
                print(f"  📊 平均RGB: [{stats['mean_rgb'][0]:.3f}, {stats['mean_rgb'][1]:.3f}, {stats['mean_rgb'][2]:.3f}]")
                print(f"  💡 亮度: {stats['brightness']:.3f}, 对比度: {stats['contrast']:.3f}")
            else:
                print(f"  ❌ 未找到有效图像")
        else:
            print(f"  ❌ 未找到用户 {user_id} 的目录")
    
    if len(user_stats) < 2:
        print(f"\n❌ 有效用户数量不足: {len(user_stats)}")
        return False
    
    print(f"\n" + "=" * 60)
    print(f"🔍 用户间差异分析")
    
    # 两两比较所有用户
    user_ids = list(user_stats.keys())
    comparison_results = []
    
    for i in range(len(user_ids)):
        for j in range(i + 1, len(user_ids)):
            user1_id = user_ids[i]
            user2_id = user_ids[j]
            
            result = compare_users(
                user_stats[user1_id], 
                user_stats[user2_id], 
                user1_id, 
                user2_id
            )
            result['user1_id'] = user1_id
            result['user2_id'] = user2_id
            comparison_results.append(result)
    
    # 总体分析
    print(f"\n" + "=" * 60)
    print(f"📈 总体差异分析")
    
    avg_composite = np.mean([r['composite_score'] for r in comparison_results])
    min_composite = min([r['composite_score'] for r in comparison_results])
    max_composite = max([r['composite_score'] for r in comparison_results])
    
    print(f"  综合差异分数统计:")
    print(f"    平均: {avg_composite:.4f}")
    print(f"    最小: {min_composite:.4f}")
    print(f"    最大: {max_composite:.4f}")
    
    # 难度分布
    difficulties = [r['difficulty'] for r in comparison_results]
    difficulty_counts = {d: difficulties.count(d) for d in set(difficulties)}
    
    print(f"  区分难度分布:")
    for difficulty, count in difficulty_counts.items():
        print(f"    {difficulty}: {count} 对用户")
    
    # 最终结论
    print(f"\n🎯 结论和建议:")
    
    if avg_composite < 0.03:
        print(f"  🚨 用户间差异极小 (平均分数: {avg_composite:.4f})")
        print(f"     这解释了为什么验证失败！")
        print(f"  💡 建议:")
        print(f"     1. 使用极高的指导强度 (30-50)")
        print(f"     2. 降低条件dropout到0.05或更低")
        print(f"     3. 考虑增加更多区分性特征")
        print(f"     4. 或者接受这是数据本身的限制")
    elif avg_composite < 0.08:
        print(f"  ❌ 用户间差异较小 (平均分数: {avg_composite:.4f})")
        print(f"     这是验证困难的主要原因")
        print(f"  💡 建议:")
        print(f"     1. 使用高指导强度 (15-25)")
        print(f"     2. 降低条件dropout到0.05")
        print(f"     3. 增加训练轮数")
        print(f"     4. 使用更强的损失函数")
    else:
        print(f"  ✅ 用户间差异足够 (平均分数: {avg_composite:.4f})")
        print(f"     问题可能在模型训练或推理配置")
        print(f"  💡 建议:")
        print(f"     1. 检查用户ID映射是否正确")
        print(f"     2. 检查条件编码器权重")
        print(f"     3. 验证推理时的条件传递")
    
    return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="分析用户间差异")
    parser.add_argument("--data_root", type=str, required=True, help="数据根目录")
    parser.add_argument("--user_ids", type=int, nargs='+', help="指定要分析的用户ID")
    parser.add_argument("--max_samples", type=int, default=20, help="每个用户最大样本数")
    
    args = parser.parse_args()
    
    success = analyze_user_differences(
        data_root=args.data_root,
        target_user_ids=args.user_ids,
        max_samples=args.max_samples
    )
    
    if success:
        print(f"\n✅ 分析完成")
    else:
        print(f"\n❌ 分析失败")

if __name__ == "__main__":
    main()
