#!/usr/bin/env python3
"""
专门针对微多普勒时频图的分析
分析用户间的步态特征差异
"""

import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

def analyze_micro_doppler_characteristics():
    """分析微多普勒时频图的特征"""
    print("🎯 微多普勒时频图特征分析")
    print("=" * 50)
    
    print("📊 微多普勒时频图特点:")
    print("  1. 时间-频率域表示")
    print("  2. 反映人体步态的多普勒频移")
    print("  3. 主要差异在步态模式的细微变化")
    print("  4. 躯干、手臂、腿部运动的频率特征")
    
    print("\n🔍 用户间差异来源:")
    print("  1. 步频差异 (walking cadence)")
    print("  2. 步幅差异 (stride length)")
    print("  3. 手臂摆动模式")
    print("  4. 身高体重影响的频率分布")
    print("  5. 个人步态习惯")
    
    print("\n⚠️  微多普勒数据的挑战:")
    print("  1. 用户间差异确实较小")
    print("  2. 但差异是有意义的生物特征")
    print("  3. 需要精细的特征学习")
    print("  4. 对噪声敏感")
    
    return True

def analyze_frequency_domain_differences(data_root: str, user_ids: list = [1, 2]):
    """分析频域差异"""
    print(f"\n🔬 分析频域特征差异")
    
    def load_and_analyze_spectrogram(user_id):
        # 查找用户目录
        from validation.analyze_user_differences import find_user_directory, load_and_analyze_images
        
        user_dir = find_user_directory(data_root, user_id)
        if not user_dir:
            return None
        
        images = load_and_analyze_images(user_dir, max_samples=10)
        if len(images) == 0:
            return None

        print(f"    加载了 {len(images)} 张图像，形状: {images.shape}")

        # 分析频域特征
        mean_img = np.mean(images, axis=0)
        print(f"    平均图像形状: {mean_img.shape}")

        # 确保图像是2D的（灰度图或RGB的一个通道）
        if len(mean_img.shape) == 3:
            # 如果是RGB图像，转换为灰度
            mean_img = np.mean(mean_img, axis=2)
            print(f"    转换为灰度图，新形状: {mean_img.shape}")

        # 计算频率轴和时间轴的能量分布
        if len(mean_img.shape) == 2:
            freq_profile = np.mean(mean_img, axis=1)  # 沿时间轴平均
            time_profile = np.mean(mean_img, axis=0)  # 沿频率轴平均
        else:
            print(f"    ⚠️  图像维度异常: {mean_img.shape}")
            return None
        
        # 计算主要频率成分
        if len(freq_profile) > 0:
            freq_peak_idx = np.argmax(freq_profile)
            freq_peak_value = freq_profile[freq_peak_idx]
        else:
            freq_peak_idx = 0
            freq_peak_value = 0.0

        # 计算频率分布的统计特征
        if len(freq_profile) > 0 and np.sum(freq_profile) > 0:
            freq_centroid = np.sum(freq_profile * np.arange(len(freq_profile))) / np.sum(freq_profile)
            freq_spread = np.sqrt(np.sum(((np.arange(len(freq_profile)) - freq_centroid) ** 2) * freq_profile) / np.sum(freq_profile))
        else:
            freq_centroid = 0.0
            freq_spread = 0.0
        
        return {
            'mean_img': mean_img,
            'freq_profile': freq_profile,
            'time_profile': time_profile,
            'freq_peak_idx': freq_peak_idx,
            'freq_peak_value': freq_peak_value,
            'freq_centroid': freq_centroid,
            'freq_spread': freq_spread,
            'num_images': len(images)
        }
    
    # 分析每个用户
    user_features = {}
    for user_id in user_ids:
        print(f"  分析用户 {user_id}...")
        features = load_and_analyze_spectrogram(user_id)
        if features:
            user_features[user_id] = features
            print(f"    ✅ 频率重心: {features['freq_centroid']:.2f}")
            print(f"    ✅ 频率扩散: {features['freq_spread']:.2f}")
            print(f"    ✅ 峰值位置: {features['freq_peak_idx']}")
        else:
            print(f"    ❌ 无法加载用户 {user_id} 数据")
    
    if len(user_features) < 2:
        print("❌ 有效用户数据不足")
        return False
    
    # 比较用户间差异
    print(f"\n📊 用户间频域差异:")
    user_list = list(user_features.keys())
    
    for i in range(len(user_list)):
        for j in range(i + 1, len(user_list)):
            user1, user2 = user_list[i], user_list[j]
            f1, f2 = user_features[user1], user_features[user2]
            
            # 频率重心差异
            centroid_diff = abs(f1['freq_centroid'] - f2['freq_centroid'])
            
            # 频率扩散差异
            spread_diff = abs(f1['freq_spread'] - f2['freq_spread'])
            
            # 峰值位置差异
            peak_diff = abs(f1['freq_peak_idx'] - f2['freq_peak_idx'])
            
            # 频率分布相关性
            correlation = np.corrcoef(f1['freq_profile'], f2['freq_profile'])[0, 1]
            
            print(f"  用户 {user1} vs 用户 {user2}:")
            print(f"    频率重心差异: {centroid_diff:.3f}")
            print(f"    频率扩散差异: {spread_diff:.3f}")
            print(f"    峰值位置差异: {peak_diff} 像素")
            print(f"    频率分布相关性: {correlation:.4f}")
            
            # 判断差异程度
            if centroid_diff < 2.0 and spread_diff < 1.0 and correlation > 0.95:
                print(f"    🚨 频域差异极小，扩散模型很难学习")
            elif centroid_diff < 5.0 and spread_diff < 3.0 and correlation > 0.9:
                print(f"    ⚠️  频域差异较小，需要强化学习")
            else:
                print(f"    ✅ 频域差异可检测")
    
    return True

def recommend_training_strategy():
    """推荐训练策略"""
    print(f"\n💡 微多普勒数据的训练策略建议")
    print("=" * 50)
    
    print("🎯 不需要重新训练VAE的情况:")
    print("  1. VAE已经能很好地重建时频图")
    print("  2. 潜在空间保持了频域信息")
    print("  3. 重建质量满足要求")
    
    print("\n🎯 不需要重新训练扩散模型的情况:")
    print("  1. 条件dropout ≤ 0.1")
    print("  2. 训练时间足够长")
    print("  3. 损失收敛良好")
    
    print("\n🔧 优先尝试的优化策略:")
    print("  1. 极高指导强度 (30-50)")
    print("  2. 更多推理步数 (100-200)")
    print("  3. 更精细的分类器 (更多数据、更多轮数)")
    print("  4. 集成多个分类器")
    
    print("\n⚠️  需要重新训练的信号:")
    print("  1. 极端参数下验证成功率仍 <30%")
    print("  2. 条件编码器嵌入相似度 >0.95")
    print("  3. 生成图像完全无差异")
    
    print("\n🚀 建议的测试顺序:")
    print("  1. 先用极端参数测试 (guidance_scale=35-50)")
    print("  2. 分析条件编码器嵌入差异")
    print("  3. 如果仍失败，考虑重新训练")
    print("  4. 重新训练时降低condition_dropout到0.02")

def create_micro_doppler_test_config():
    """创建微多普勒专用测试配置"""
    print(f"\n⚙️  微多普勒专用测试配置")
    print("=" * 50)
    
    configs = {
        "conservative": {
            "guidance_scale": 25.0,
            "num_inference_steps": 100,
            "classifier_epochs": 40,
            "classifier_lr": 1e-4,
            "description": "保守测试 - 适合初次验证"
        },
        "aggressive": {
            "guidance_scale": 40.0,
            "num_inference_steps": 150,
            "classifier_epochs": 60,
            "classifier_lr": 5e-5,
            "description": "激进测试 - 强化微小差异"
        },
        "extreme": {
            "guidance_scale": 50.0,
            "num_inference_steps": 200,
            "classifier_epochs": 80,
            "classifier_lr": 2e-5,
            "description": "极端测试 - 最大化特征保持"
        }
    }
    
    for name, config in configs.items():
        print(f"\n📋 {config['description']} ({name}):")
        for key, value in config.items():
            if key != 'description':
                print(f"  {key}: {value}")
    
    return configs

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="微多普勒时频图分析")
    parser.add_argument("--action", choices=["analyze", "frequency", "recommend"], required=True)
    parser.add_argument("--data_root", type=str, help="数据根目录")
    parser.add_argument("--user_ids", type=int, nargs='+', default=[1, 2], help="用户ID列表")
    
    args = parser.parse_args()
    
    if args.action == "analyze":
        analyze_micro_doppler_characteristics()
        create_micro_doppler_test_config()
        
    elif args.action == "frequency":
        if not args.data_root:
            print("❌ 需要提供数据根目录")
            return
        analyze_frequency_domain_differences(args.data_root, args.user_ids)
        
    elif args.action == "recommend":
        recommend_training_strategy()

if __name__ == "__main__":
    main()
